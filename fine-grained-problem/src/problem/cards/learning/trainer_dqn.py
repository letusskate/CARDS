import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import random
import time
import sys
import numpy as np
import torch

from src.problem.cards.environment.environment import Environment
from src.problem.cards.learning.brain_dqn import BrainDQN
from src.problem.cards.environment.cards import CardsProblem
from src.util.prioritized_replay_memory import PrioritizedReplayMemory

MEMORY_CAPACITY = 50000
GAMMA = 1
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100
MIN_VAL = -1000000


class TrainerDQN:
    def __init__(self, args):
        self.args = args
        np.random.seed(self.args.seed)
        self.n_feat = (
            len(args.feature_names) if hasattr(args, "feature_names") else 16
        )  # 可以自定义，原先是1，后改变make_nn_input后设置为13，变大点变成16
        self.n_action = (
            args.n_action if hasattr(args, "n_action") else 11
        )  # z的最大取值+1 #wyb set to 11，但不清楚应该是多大
        # ### wyb test n_action
        # print("naction!!!:",self.n_action)

        self.reward_scaling = 1.0
        self.validation_set = CardsProblem.generate_dataset(
            size=VALIDATION_SET_SIZE,
            n_L=args.n_L,
            n_N=args.n_N,
            n_O=args.n_O,
            seed=np.random.randint(10000),
        )
        self.brain = BrainDQN(self.args, self.n_feat)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)
        self.steps_done = 0
        self.init_memory_counter = 0

        if args.n_step == -1:
            self.n_step = args.n_L * args.n_N * args.n_N * args.n_O
        else:
            self.n_step = self.args.n_step

        print("***********************************************************")
        print("[INFO] n_feat: %d" % self.n_feat)
        print("***********************************************************")

    def run_training(self):
        start_time = time.time()
        if self.args.plot_training:
            iter_list = []
            reward_list = []

        self.initialize_memory()
        print("[INFO]", "iter", "time", "avg_reward_learning", "loss", "beta")
        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):
            loss, beta = self.run_episode(i, memory_initialization=False)
            if (i % 10 == 0 and i < 101) or i % 100 == 0:
                avg_reward = 0.0
                for j in range(len(self.validation_set)):
                    avg_reward += self.evaluate_instance(j) / self.reward_scaling
                avg_reward = avg_reward / len(self.validation_set)
                cur_time = round(time.time() - start_time, 2)
                print("[DATA]", i, cur_time, avg_reward, loss, beta)
                sys.stdout.flush()
                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()
                    plt.plot(
                        iter_list, reward_list, linestyle="-", label="DQN", color="y"
                    )
                    plt.legend(loc=3)
                    out_file = "%s/training_curve_reward.png" % self.args.save_dir
                    plt.savefig(out_file)
                fn = "iter_%d_model.pth" % i
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn)
                elif i % 10000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn)

    def initialize_memory(self):
        while self.init_memory_counter < MEMORY_CAPACITY:
            self.run_episode(0, memory_initialization=True)
        print("[INFO] Memory Initialized")

    def run_episode(self, episode_idx, memory_initialization):
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
        set_list = []
        max_step = np.prod(
            [len(instance.L), len(instance.N), len(instance.N), len(instance.O)]
        )
        rewards_vector = np.zeros(max_step)
        actions_vector = np.zeros(max_step, dtype=np.int16)
        available_vector = np.zeros((max_step, self.n_action))
        idx = 0
        total_loss = 0
        temperature = max(
            0.0,
            min(
                self.args.max_softmax_beta,
                (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta,
            ),
        )
        while True:
            nn_input = env.make_nn_input(cur_state, self.args.mode)
            # ### test wyb nninput shape
            # print(f"step {idx}, nn_input.shape: {nn_input.shape}")
            avail = np.zeros(self.n_action)
            valid_actions = env.get_valid_actions(cur_state)
            for a in valid_actions:
                assert (
                    0 <= a < self.n_action
                ), f"Action {a} out of range for n_action={self.n_action}"  ## wyb 限制z，小于n_action
                avail[a] = 1
            avail_idx = np.argwhere(avail == 1).reshape(-1)
            if len(avail_idx) == 0:
                break
            if memory_initialization:
                action = random.choice(avail_idx)
            else:
                action = self.soft_select_action(nn_input, avail, temperature)
                self.steps_done += 1
                if self.steps_done % UPDATE_TARGET_FREQUENCY == 0:
                    self.brain.update_target_model()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)
            set_list.append(nn_input)
            rewards_vector[idx] = reward
            actions_vector[idx] = action
            available_vector[idx] = avail
            if cur_state.is_done():
                break
            idx += 1
        episode_last_idx = idx
        for i in range(max_step):
            if i <= episode_last_idx:
                cur_set = set_list[i]
                cur_available = available_vector[i]
            else:
                cur_set = set_list[episode_last_idx]
                cur_available = available_vector[episode_last_idx]
            if i + self.n_step < max_step:
                next_set = set_list[i + self.n_step]
                next_available = available_vector[i + self.n_step]
            else:
                next_set = torch.FloatTensor(np.zeros_like(set_list[0]))
                next_available = cur_available
                if self.args.mode == "gpu":
                    next_set = next_set.cuda()
            state_features = (cur_set, cur_available)
            next_state_features = (next_set, next_available)
            reward = sum(rewards_vector[i : i + self.n_step])
            action = actions_vector[i]
            sample = (state_features, action, reward, next_state_features)
            if memory_initialization:
                error = abs(reward)
                self.init_memory_counter += 1
                step_loss = 0
            else:
                x, y, errors = self.get_targets([(0, sample, 0)])
                error = errors[0]
                step_loss = self.learning()
            self.memory.add(error, sample)
            total_loss += step_loss
        return total_loss, temperature

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
            ## wyb test input shape
            # print("!!!!!!!!nn_input shape:", nn_input.shape)
            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = np.zeros(self.n_action)
            valid_actions = env.get_valid_actions(cur_state)
            for a in valid_actions:
                assert (
                    0 <= a < self.n_action
                ), f"Action {a} out of range for n_action={self.n_action}"  ## wyb 限制z，小于n_action
                avail[a] = 1
            action = self.select_action(nn_input, avail)
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)
            total_reward += reward
            if cur_state.is_done():
                break
        return total_reward

    def select_action(self, set_input, available):
        batched_set = set_input.unsqueeze(0)
        available = available.astype(bool)
        out = self.brain.predict(batched_set, target=False).squeeze(0)
        action_idx = np.argmax(out[available])
        action = np.arange(len(out))[available][action_idx]
        return action

    def soft_select_action(self, set_input, available, beta):
        batched_set = set_input.unsqueeze(0)
        available = available.astype(bool)
        out = self.brain.predict(batched_set, target=False)[0].reshape(-1)
        if len(out[available]) > 1:
            logits = out[available] - out[available].mean()
            div = ((logits**2).sum() / (len(logits) - 1)) ** 0.5
            logits = logits / div
            probabilities = np.exp(beta * logits)
            norm = probabilities.sum()
            if norm == np.inf:
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action
            probabilities /= norm
        else:
            probabilities = [1]
        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        action = np.arange(len(out))[available][action_idx]
        return action

    def get_targets(self, batch):
        batch_len = len(batch)
        set_input, avail = list(zip(*[e[1][0] for e in batch]))
        set_batch = torch.stack(set_input)
        next_set_input, next_avail = list(zip(*[e[1][3] for e in batch]))
        next_set_batch = torch.stack(next_set_input)
        p = self.brain.predict(set_batch, target=False)
        p_ = self.brain.predict(next_set_batch, target=False)
        p_target_ = self.brain.predict(next_set_batch, target=True)
        x = []
        y = []
        errors = np.zeros(len(batch))
        for i in range(batch_len):
            sample = batch[i][1]
            state_set, state_avail = sample[0]
            action = sample[1]
            reward = sample[2]
            _, next_state_avail = sample[3]
            next_action_indices = np.argwhere(next_state_avail == 1).reshape(-1)
            t = p[i]
            q_value_prediction = t[action]
            if len(next_action_indices) == 0:
                td_q_value = reward
                t[action] = td_q_value
            else:
                mask = np.zeros(p_[i].shape, dtype=bool)
                mask[next_action_indices] = True
                best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action = np.arange(len(mask))[mask.reshape(-1)][
                    best_valid_next_action_id
                ]
                td_q_value = reward + GAMMA * p_target_[i][best_valid_next_action]
                t[action] = td_q_value
            state = (state_set, state_avail)
            x.append(state)
            y.append(t)
            errors[i] = abs(q_value_prediction - td_q_value)
        return x, y, errors

    def learning(self):
        batch = self.memory.sample(self.args.batch_size)
        x, y, errors = self.get_targets(batch)
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
        loss = self.brain.train(x, y)
        return round(loss, 4)
