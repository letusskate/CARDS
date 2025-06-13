import numpy as np
from src.problem.cards.environment.cards import CardsProblem
from src.problem.cards.environment.state import CardsState
import torch


class Environment:
    def __init__(self, n_action, instance, n_feat, reward_scaling=1.0):
        """
        instance: CardsProblem instance
        reward_scaling: scaling factor of reward
        """
        self.instance = instance
        self.reward_scaling = reward_scaling
        self.n_feat = n_feat
        self.n_action = n_action  # for example the maximum allowable value of z is 1 such as 11

    def get_initial_environment(self):
        """
         return initial state
        """
        return CardsState(self.instance)

    # ### variable length nn input
    #     def make_nn_input(self, cur_state, mode='cpu'):
    #         """
    #         construct neural network input return shape n action n feat each line is a feature of an action and is self n feat in length
    #         """
    #         valid_actions = cur_state.get_valid_actions()
    #         n_action = len(valid_actions)
    #         feat_list = []
    #         # can customize the features according to your actual needs such as assign idx action value current value of z relevant static parameters etc
    #         l, n, m, o = cur_state.get_assign_indices()  
    #         for a in valid_actions:
    #             # examples of static parameters
    #             lambda_ln = self.instance.lambda_ln[l, n]
    #             a_lnm = self.instance.a_lnm[l, n, m]
    #             x_lm = self.instance.x_lm[l, m]
    #             omega_l = self.instance.omega_l[l]
    #             W_m = self.instance.W_m[m]
    #             kappa_l = self.instance.kappa_l[l]
    #             nu_nmo = self.instance.nu_nmo[n, m, o]
    #             # the current z value
    #             z_val = cur_state.z[l, n, m, o]
    #             # action normalization
    #             a_scalar = a / 10.0
    #             # stitching features
    #             feat = np.array([
    #                 l / len(self.instance.L),
    #                 n / len(self.instance.N),
    #                 m / len(self.instance.N),
    #                 o / len(self.instance.O),
    #                 a_scalar,
    #                 lambda_ln, a_lnm, x_lm, omega_l, W_m, kappa_l, nu_nmo,
    #                 z_val
    #             ], dtype=np.float32)
    #             # zeroing or truncation to self n feat
    #             if len(feat) < self.n_feat:
    #                 feat = np.pad(feat, (0, self.n_feat - len(feat)), 'constant')
    #             else:
    #                 feat = feat[:self.n_feat]
    #             feat_list.append(feat)
    #         feat_arr = np.stack(feat_list, axis=0)  # [n_action, n_feat]
    #         feat_tensor = torch.FloatTensor(feat_arr)
    #         if mode == 'gpu':
    #             feat_tensor = feat_tensor.cuda()
    #         return feat_tensor

    ### fixed output length of n action nn input
    def make_nn_input(self, cur_state, mode="cpu"):
        """
        construct neural network input return shape n action n feat each row is a feature of an action and is self n feat in length
        """
        n_action = self.n_action  # for example the maximum allowable value of z is 1 such as 11
        feat_list = []
        l, n, m, o = cur_state.get_assign_indices()  # this approach needs to be implemented
        valid_actions = set(cur_state.get_valid_actions())
        for a in range(n_action):
            if a in valid_actions:
                # construct real features
                lambda_ln = self.instance.lambda_ln[l, n]
                a_lnm = self.instance.a_lnm[l, n, m]
                x_lm = self.instance.x_lm[l, m]
                omega_l = self.instance.omega_l[l]
                W_m = self.instance.W_m[m]
                kappa_l = self.instance.kappa_l[l]
                nu_nmo = self.instance.nu_nmo[n, m, o]
                z_val = cur_state.z[l, n, m, o]
                a_scalar = a / (n_action - 1) if n_action > 1 else 0
                feat = np.array(
                    [
                        l / len(self.instance.L),
                        n / len(self.instance.N),
                        m / len(self.instance.N),
                        o / len(self.instance.O),
                        a_scalar,
                        lambda_ln,
                        a_lnm,
                        x_lm,
                        omega_l,
                        W_m,
                        kappa_l,
                        nu_nmo,
                        z_val,
                    ],
                    dtype=np.float32,
                )
                if len(feat) < self.n_feat:
                    feat = np.pad(feat, (0, self.n_feat - len(feat)), "constant")
                else:
                    feat = feat[: self.n_feat]
            else:
                feat = np.zeros(self.n_feat, dtype=np.float32)  # or all -1
            feat_list.append(feat)
        feat_arr = np.stack(feat_list, axis=0)  # [n_action, n_feat]
        feat_tensor = torch.FloatTensor(feat_arr)
        if mode == "gpu":
            feat_tensor = feat_tensor.cuda()
        return feat_tensor

    def get_next_state_with_reward(self, cur_state, action):
        """
        perform actions to return to new statuses and rewards
        """
        new_state = cur_state.step(action)
        reward = 0
        if new_state.is_done():
            # only the final state calculates the objective function as a reward
            # obj = sum(z[l, n, m, o]) which can be adjusted according to the objective function
            reward = np.sum(new_state.z)
            reward = reward * self.reward_scaling
        return new_state, reward

    def get_valid_actions(self, cur_state):
        """
        returns all optional actions in the current state of the assign idx position the feasible value of z
        """
        return cur_state.get_valid_actions()
