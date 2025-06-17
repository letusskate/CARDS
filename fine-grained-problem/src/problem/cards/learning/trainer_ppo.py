import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import time
import sys
import numpy as np
import torch

from src.problem.cards.environment.cards import CardsProblem
from src.problem.cards.environment.environment import Environment
from src.problem.cards.learning.brain_ppo import BrainPPO
from src.util.replay_memory import ReplayMemory

VALIDATION_SET_SIZE = 100
MIN_VAL = -1000000


class TrainerPPO:
    def __init__(self, args):
        self.args = args
        np.random.seed(self.args.seed)
        self.n_feat = len(args.feature_names) if hasattr(args, "feature_names") else 16
        self.n_action = (
            args.n_action if hasattr(args, "n_action") else 11
        )  # z的最大取值+1 #wyb set to 11，但不清楚应该是多大
        self.reward_scaling = 1  # default 1, wyb change to o.1
        self.validation_set = CardsProblem.generate_dataset(
            size=VALIDATION_SET_SIZE,
            n_L=args.n_L,
            n_N=args.n_N,
            n_O=args.n_O,
            seed=np.random.randint(10000),
        )
        self.brain = BrainPPO(self.args, self.n_feat)
        self.memory = ReplayMemory()
        self.time_step = 0
        print("***********************************************************")
        print("[INFO] n_feat: %d" % self.n_feat)
        print("***********************************************************")

    def run_training(self):
        start_time = time.time()
        if self.args.plot_training:
            iter_list = []
            reward_list = []
        print("[INFO]", "iter", "time", "avg_reward_learning")
        cur_best_reward = MIN_VAL
        for i in range(self.args.n_episode):
            self.run_episode()
            if (i % 10 == 0 and i < 101) or i % 100 == 0:
                avg_reward = 0.0
                for j in range(len(self.validation_set)):
                    avg_reward += self.evaluate_instance(j) / self.reward_scaling
                avg_reward = avg_reward / len(self.validation_set)
                cur_time = round(time.time() - start_time, 2)
                print("[DATA]", i, cur_time, avg_reward)
                sys.stdout.flush()
                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()
                    plt.plot(
                        iter_list, reward_list, linestyle="-", label="PPO", color="y"
                    )
                    plt.legend(loc=3)
                    out_file = "%s/training_curve_reward.png" % self.args.save_dir
                    plt.savefig(out_file)
                    plt.clf()
                fn = "iter_%d_model.pth" % i
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(self.args.save_dir, fn)
                elif i % 10000 == 0:
                    self.brain.save(self.args.save_dir, fn)

    def run_episode(self):
        instance = CardsProblem.generate_random_instance(
            n_L=self.args.n_L, n_N=self.args.n_N, n_O=self.args.n_O, seed=-1
        )
        env = Environment(
            n_action=self.n_action,
            instance=instance,
            n_feat=self.n_feat,
            reward_scaling=self.reward_scaling,
        )
        cur_state = env.get_initial_environment()
        while True:
            self.time_step += 1
            nn_input = env.make_nn_input(cur_state, self.args.mode)
            valid_actions = env.get_valid_actions(cur_state)
            avail = np.zeros(self.args.n_action)
            for a in valid_actions:
                assert (
                    0 <= a < self.n_action
                ), f"Action {a} out of range for n_action={self.n_action}"  ## wyb 限制z，小于n_action
                avail[a] = 1
            available_tensor = torch.FloatTensor(avail)
            # print("!!!!!!!!nn_input shape:", nn_input.shape)
            out_action, log_prob_action, _ = self.brain.policy_old.act(
                nn_input, available_tensor
            )
            action = out_action.item()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)
            # # wyb只存储被选中动作的特征，后来发现这样无法收敛
            # state_feature = nn_input[action].unsqueeze(0)  # shape: [1, n_feat]
            # self.memory.add_sample(state_feature, out_action, log_prob_action, reward, cur_state.is_done(), available_tensor)
            self.memory.add_sample(
                nn_input,
                out_action,
                log_prob_action,
                reward,
                cur_state.is_done(),
                available_tensor,
            )
            if self.time_step % self.args.update_timestep == 0:
                self.brain.update(self.memory)
                self.memory.clear_memory()
                self.time_step = 0
            if cur_state.is_done():
                break

    def evaluate_instance(self, idx):
        instance = self.validation_set[idx]
        env = Environment(
            n_action=self.n_action,
            instance=instance,
            n_feat=self.n_feat,
            reward_scaling=self.reward_scaling,
        )
        cur_state = env.get_initial_environment()
        total_reward = 0
        while True:
            nn_input = env.make_nn_input(cur_state, self.args.mode)
            valid_actions = env.get_valid_actions(cur_state)
            avail = np.zeros(self.args.n_action)
            for a in valid_actions:
                assert (
                    0 <= a < self.n_action
                ), f"Action {a} out of range for n_action={self.n_action}"  ## wyb 限制z，小于n_action
                avail[a] = 1
            available_tensor = torch.FloatTensor(avail)
            out_action, _, _ = self.brain.policy_old.act(nn_input, available_tensor)
            action = out_action.item()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)
            total_reward += reward
            if cur_state.is_done():
                break
        return total_reward
