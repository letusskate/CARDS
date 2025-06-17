#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/search.hh>
#include <gecode/minimodel.hh>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <map>
#include <limits>
#include <string>
#include <cxxopts.hpp>

using namespace Gecode;
using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

std::default_random_engine generator(0);

class OptionsCards: public Options {
public:
    int n_L, n_N, n_O, seed;
    float temperature;
    bool cache;
    std::string model;
    OptionsCards(const char* s, int n_L, int n_N, int n_O, int seed, float temperature, bool cache, std::string model)
        : Options(s), n_L(n_L), n_N(n_N), n_O(n_O), seed(seed), temperature(temperature), cache(cache), model(model) {}
};

class Cards_DP : public IntMaximizeScript {
public:
    IntVarArray z;
    py::object to_run;
    py::object py_instance;
    int n_L, n_N, n_O, seed;
    float temperature;
    bool use_cache;
    std::string rl_algorithm;

    // 参数
    std::vector<int> lambda_ln, a_lnm, x_lm, omega_l, W_m, kappa_l, nu_nmo, K_p_cap, d_nmop, K_cap;

    // 多维索引转一维
    int idx(int l, int n, int m, int o) const {
        return ((l * n_N + n) * n_N + m) * n_O + o;
    }

    Cards_DP(const OptionsCards& opt) : IntMaximizeScript(opt) {
        n_L = opt.n_L;
        n_N = opt.n_N;
        n_O = opt.n_O;
        seed = opt.seed;
        temperature = opt.temperature;
        use_cache = opt.cache;
        rl_algorithm = opt.model;

        // Python binding
        string model_folder = "./selected-models/" + rl_algorithm + "/cards/n-L-" + std::to_string(n_L) +
                              "/n-N-" + std::to_string(n_N) + "/n-O-" + std::to_string(n_O) + "/seed-" + std::to_string(seed);

        auto to_run_module = py::module::import("src.problem.cards.solving.solver_binding");
        to_run = to_run_module.attr("SolverBinding")(model_folder, n_L, n_N, n_O, seed, rl_algorithm);
        py_instance = to_run.attr("get_instance")();

        // 读取参数
        lambda_ln = py_instance.attr("lambda_ln").cast<std::vector<int>>();
        a_lnm     = py_instance.attr("a_lnm").cast<std::vector<int>>();
        x_lm      = py_instance.attr("x_lm").cast<std::vector<int>>();
        omega_l   = py_instance.attr("omega_l").cast<std::vector<int>>();
        W_m       = py_instance.attr("W_m").cast<std::vector<int>>();
        kappa_l   = py_instance.attr("kappa_l").cast<std::vector<int>>();
        nu_nmo    = py_instance.attr("nu_nmo").cast<std::vector<int>>();
        K_p_cap   = py_instance.attr("K_p_cap").cast<std::vector<int>>();
        d_nmop    = py_instance.attr("d_nmop").cast<std::vector<int>>();
        K_cap     = py_instance.attr("K_cap").cast<std::vector<int>>();

        int n_var = n_L * n_N * n_N * n_O;
        z = IntVarArray(*this, n_var, 0, 100); // 假设z最大100

        // 约束1: sum_{m,o} z_{l,n,m,o} ≤ lambda_ln[l, n]
        for (int l = 0; l < n_L; ++l)
            for (int n = 0; n < n_N; ++n) {
                IntVarArgs sum_mo;
                for (int m = 0; m < n_N; ++m)
                    for (int o = 0; o < n_O; ++o)
                        sum_mo << z[idx(l, n, m, o)];
                rel(*this, sum(sum_mo) <= lambda_ln[l * n_N + n]);
            }

        // 约束2: sum_{o} z_{l,n,m,o} ≤ a_{lnm} x_{lm} lambda_{ln}
        for (int l = 0; l < n_L; ++l)
            for (int n = 0; n < n_N; ++n)
                for (int m = 0; m < n_N; ++m) {
                    IntVarArgs sum_o;
                    for (int o = 0; o < n_O; ++o)
                        sum_o << z[idx(l, n, m, o)];
                    int a = a_lnm[((l * n_N + n) * n_N) + m];
                    int x = x_lm[l * n_N + m];
                    int lam = lambda_ln[l * n_N + n];
                    rel(*this, sum(sum_o) <= a * x * lam);
                }

        // 约束3: sum_{l} omega_l * sum_{n,o} z_{l,n,m,o} ≤ W_m[m]
        for (int m = 0; m < n_N; ++m) {
            LinearExpr total = 0;
            for (int l = 0; l < n_L; ++l) {
                LinearExpr sum_no = 0;
                for (int n = 0; n < n_N; ++n)
                    for (int o = 0; o < n_O; ++o)
                        sum_no += z[idx(l, n, m, o)];
                total += omega_l[l] * sum_no;
            }
            rel(*this, total <= W_m[m]);
        }

        // 约束4: sum_{l} z_{l,n,m,o} * kappa_l[l] ≤ nu_nmo[n, m, o]
        for (int n = 0; n < n_N; ++n)
            for (int m = 0; m < n_N; ++m)
                for (int o = 0; o < n_O; ++o) {
                    LinearExpr sum_l = 0;
                    for (int l = 0; l < n_L; ++l)
                        sum_l += z[idx(l, n, m, o)] * kappa_l[l];
                    rel(*this, sum_l <= nu_nmo[(n * n_N + m) * n_O + o]);
                }

        // 约束5: sum_{l} kappa_l[l] * sum_{m,o} z_{l,n,m,o} ≤ K_p_cap[n]
        for (int n = 0; n < n_N; ++n) {
            LinearExpr total = 0;
            for (int l = 0; l < n_L; ++l) {
                LinearExpr sum_mo = 0;
                for (int m = 0; m < n_N; ++m)
                    for (int o = 0; o < n_O; ++o)
                        sum_mo += z[idx(l, n, m, o)];
                total += kappa_l[l] * sum_mo;
            }
            rel(*this, total <= K_p_cap[n]);
        }

        // 约束6: sum_{l} kappa_l[l] * sum_{n,m,o} z_{l,n,m,o} * d_nmop[n, m, o, p] ≤ K_cap[p]
        for (int p = 0; p < n_N; ++p) {
            LinearExpr total = 0;
            for (int l = 0; l < n_L; ++l)
                for (int n = 0; n < n_N; ++n)
                    for (int m = 0; m < n_N; ++m)
                        for (int o = 0; o < n_O; ++o)
                            total += z[idx(l, n, m, o)] * kappa_l[l] * d_nmop[(((n * n_N + m) * n_O + o) * n_N) + p];
            rel(*this, total <= K_cap[p]);
        }

        // 分支策略
        if (rl_algorithm == "dqn") {
            branch(*this, z, INT_VAR_NONE(), INT_VAL(&value_selector));
        } else if (rl_algorithm == "ppo") {
            branch(*this, z, INT_VAR_NONE(), INT_VAL(&probability_selector));
        }
    }

    // 构造当前状态和可行动作
    void get_state_and_avail(int assign_idx, py::object& cur_state, std::vector<double>& avail) const {
        // 构造当前状态
        py::object CardsState = py::module::import("src.problem.cards.environment.state").attr("CardsState");
        // z_vec: 当前变量赋值
        std::vector<int> z_vec(z.size());
        for (int i = 0; i < z.size(); ++i) z_vec[i] = z[i].val();
        cur_state = CardsState(py_instance, py::cast(z_vec), assign_idx);
        // 获取可行动作
        py::list valid_actions = cur_state.attr("get_valid_actions")();
        int n_action = valid_actions.attr("__len__")();
        avail = std::vector<double>(n_action, 0.0);
        for (int i = 0; i < n_action; ++i) {
            int a = valid_actions[i].cast<int>();
            if (a >= 0 && a < n_action) avail[a] = 1.0;
        }
    }

    static int value_selector(const Space& home, IntVar x, int i) {
        const Cards_DP& cards_dp = static_cast<const Cards_DP&>(home);
        py::object cur_state;
        std::vector<double> avail;
        cards_dp.get_state_and_avail(i, cur_state, avail);
        std::vector<double> q_values = cards_dp.to_run.attr("predict_dqn")(cur_state, avail).cast<std::vector<double>>();
        int best = -1;
        double best_val = -1e9;
        for (int j = 0; j < q_values.size(); ++j) {
            if (avail[j] > 0 && q_values[j] > best_val) {
                best = j;
                best_val = q_values[j];
            }
        }
        return best;
    }

    static int probability_selector(const Space& home, IntVar x, int i) {
        const Cards_DP& cards_dp = static_cast<const Cards_DP&>(home);
        py::object cur_state;
        std::vector<double> avail;
        cards_dp.get_state_and_avail(i, cur_state, avail);
        std::vector<double> prob_values = cards_dp.to_run.attr("predict_ppo")(cur_state, avail, cards_dp.temperature).cast<std::vector<double>>();
        std::discrete_distribution<int> distribution(prob_values.begin(), prob_values.end());
        int action = distribution(generator);
        return action;
    }

    virtual IntVar cost(void) const {
        return sum(z);
    }

    virtual void print(std::ostream& os) const {
        os << "Objective: " << cost().val() << std::endl;
    }
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    cxxopts::Options options("CardsSolver", "Cards solver for DQN and PPO");
    options.add_options()
        ("model", "model to run (dqn/ppo)", cxxopts::value<std::string>()->default_value("dqn"))
        ("time", "Time limit in ms", cxxopts::value<int>()->default_value("60000"))
        ("n_L", "n_L", cxxopts::value<int>()->default_value("5"))
        ("n_N", "n_N", cxxopts::value<int>()->default_value("4"))
        ("n_O", "n_O", cxxopts::value<int>()->default_value("3"))
        ("seed", "random seed", cxxopts::value<int>()->default_value("1"))
        ("temperature", "temperature for PPO", cxxopts::value<float>()->default_value("1.0"))
        ("cache", "enable cache", cxxopts::value<bool>()->default_value("1"));
    auto result = options.parse(argc, argv);

    OptionsCards opt("Cards problem",
                     result["n_L"].as<int>(),
                     result["n_N"].as<int>(),
                     result["n_O"].as<int>(),
                     result["seed"].as<int>(),
                     result["temperature"].as<float>(),
                     result["cache"].as<bool>(),
                     result["model"].as<std::string>());

    Cards_DP* p = new Cards_DP(opt);
    BAB<Cards_DP> engine(p);
    delete p;
    while (Cards_DP* s = engine.next()) {
        s->print(std::cout);
        delete s;
    }
    return 0;
}