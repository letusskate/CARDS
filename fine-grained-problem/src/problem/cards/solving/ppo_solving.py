import sys
import os
import argparse
import torch
import numpy as np

sys.path.append(os.path.join(sys.path[0], "..", "..", "..", ".."))

from src.problem.cards.environment.environment import Environment
from src.problem.cards.environment.cards import CardsProblem
from src.problem.cards.learning.brain_ppo import BrainPPO


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_L", type=int, default=10)
    parser.add_argument("--n_N", type=int, default=8)
    parser.add_argument("--n_O", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_layer", type=int, default=2)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_action", type=int, default=11)
    parser.add_argument("--n_feat", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # 加载模型
    policy = BrainPPO(args, args.n_feat).policy
    policy.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    policy.eval()

    # 生成一个实例
    instance = CardsProblem.generate_random_instance(
        args.n_L, args.n_N, args.n_O, args.seed
    )
    env = Environment(args.n_action, instance, args.n_feat)
    cur_state = env.get_initial_environment()

    solution = []
    total_reward = 0

    while True:
        nn_input = env.make_nn_input(cur_state, "cpu")
        avail = np.zeros(args.n_action)
        valid_actions = cur_state.get_valid_actions()
        for a in valid_actions:
            avail[a] = 1
        available = torch.tensor(avail, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = policy(
                torch.tensor(nn_input).unsqueeze(0), available.unsqueeze(0)
            )
            action_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            action_probs = action_probs * avail  # mask非法动作
            action = np.argmax(action_probs)
        cur_state, reward = env.get_next_state_with_reward(cur_state, action)
        solution.append(action)
        total_reward += reward
        if cur_state.is_done():
            break

    print("SOLUTION:", solution)
    print("TOTAL REWARD:", total_reward)
