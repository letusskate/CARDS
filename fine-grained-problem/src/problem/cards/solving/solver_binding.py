import torch
import numpy as np
from types import SimpleNamespace

from src.problem.cards.environment.cards import CardsProblem
from src.problem.cards.learning.brain_dqn import BrainDQN
from src.problem.cards.learning.brain_ppo import BrainPPO


class SolverBinding(object):
    """
    Definition of the c++ and the pytorch model for cards.
    """

    def __init__(self, load_folder, n_L, n_N, n_O, seed, rl_algorithm):
        self.n_L = n_L
        self.n_N = n_N
        self.n_O = n_O
        self.seed = seed
        self.rl_algorithm = rl_algorithm
        self.load_folder = load_folder

        # 生成实例
        self.instance = CardsProblem.generate_random_instance(
            n_L=n_L, n_N=n_N, n_O=n_O, seed=seed
        )

        # 查找模型
        self.model_file, self.latent_dim, self.hidden_layer, self.n_feat = (
            self.find_model()
        )

        # 加载模型
        args = SimpleNamespace(
            latent_dim=self.latent_dim, hidden_layer=self.hidden_layer
        )
        if rl_algorithm == "dqn":
            self.model = BrainDQN(args, self.n_feat)
            self.model.model.load_state_dict(
                torch.load(self.model_file, map_location="cpu"), strict=True
            )
            self.model.model.eval()
        elif rl_algorithm == "ppo":
            self.model = BrainPPO(args, self.n_feat)
            self.model.policy.load_state_dict(
                torch.load(self.model_file, map_location="cpu"), strict=True
            )
            self.model.policy.eval()
        else:
            raise Exception("RL algorithm not implemented")

    def find_model(self):
        log_file_path = self.load_folder + "/log-training.txt"
        best_reward = -1e9
        best_it = -1
        with open(log_file_path, "r") as f:
            for line in f:
                if "[INFO]" in line:
                    line = line.split(" ")
                    if line[1] == "latent_dim:":
                        latent_dim = int(line[2].strip())
                    elif line[1] == "hidden_layer:":
                        hidden_layer = int(line[2].strip())
                    elif line[1] == "n_feat:":
                        n_feat = int(line[2].strip())
                if "[DATA]" in line:
                    line = line.split(" ")
                    it = int(line[1].strip())
                    reward = float(line[3].strip())
                    if reward > best_reward:
                        best_reward = reward
                        best_it = it
        assert best_it >= 0, "No model found"
        model_str = "%s/iter_%d_model.pth" % (self.load_folder, best_it)
        return model_str, latent_dim, hidden_layer, n_feat

    def build_state_feats(self, cur_state):
        # 必须与训练时 make_nn_input 保持一致
        from src.problem.cards.environment.environment import Environment

        env = Environment(self.n_N, self.instance, self.n_feat)
        nn_input = env.make_nn_input(cur_state, "cpu")
        return torch.FloatTensor(nn_input)

    def predict_dqn(self, cur_state, avail):
        with torch.no_grad():
            state_feat_tensor = self.build_state_feats(cur_state).unsqueeze(0)
            out = self.model.model(state_feat_tensor).cpu().numpy().squeeze(0)
            out = out * avail
        return out.tolist()

    def predict_ppo(self, cur_state, avail, temperature):
        with torch.no_grad():
            state_feat_tensor = self.build_state_feats(cur_state).unsqueeze(0)
            logits, _ = self.model.policy(
                state_feat_tensor, torch.tensor(avail).unsqueeze(0)
            )
            probs = torch.softmax(logits / temperature, dim=-1).squeeze(0).cpu().numpy()
            probs = probs * avail
        return probs.tolist()

    # 可选：暴露实例参数给C++
    def get_instance(self):
        return self.instance
