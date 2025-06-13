import numpy as np


class CardsState:
    def __init__(self, instance, z=None, assign_idx=0):
        """
        instance: CardsProblem case
        z: assignment of all current z variables(shape: |L| x |N| x |N| x |O|),the unassigned value is 0
        assign_idx: index of the variable to be assigned linear expansion
        """
        self.instance = instance
        if z is None:
            self.z = np.zeros(
                (len(instance.L), len(instance.N), len(instance.N), len(instance.O)),
                dtype=int,
            )
        else:
            self.z = z.copy()
        self.assign_idx = assign_idx  # it is used for sequential assignment and the action space can also be customized

    def step(self, action):
        """
        action: assign value to the z variable pointed to by the current assign_idx (e.g. 0,1,2,...)
        returns new state
        """
        new_z = self.z.copy()
        idx = np.unravel_index(self.assign_idx, new_z.shape)
        new_z[idx] = action
        return CardsState(self.instance, new_z, self.assign_idx + 1)

    def is_done(self):
        """
        whether all variables have been assigned
        """
        return self.assign_idx >= self.z.size

    def is_success(self):
        """
        whether the current solution satisfies all constraints
        """
        # Constraint 1
        for l in range(len(self.instance.L)):
            for n in range(len(self.instance.N)):
                if np.sum(self.z[l, n, :, :]) > self.instance.lambda_ln[l, n]:
                    return False
        # Constraint 2
        for l in range(len(self.instance.L)):
            for n in range(len(self.instance.N)):
                for m in range(len(self.instance.N)):
                    if (
                        np.sum(self.z[l, n, m, :])
                        > self.instance.a_lnm[l, n, m]
                        * self.instance.x_lm[l, m]
                        * self.instance.lambda_ln[l, n]
                    ):
                        return False
        # Constraint 3
        for m in range(len(self.instance.N)):
            total = 0
            for l in range(len(self.instance.L)):
                for n in range(len(self.instance.N)):
                    total += self.instance.omega_l[l] * np.sum(self.z[l, n, m, :])
            if total > self.instance.W_m[m]:
                return False
        # Constraint 4
        for n in range(len(self.instance.N)):
            for m in range(len(self.instance.N)):
                for o in range(len(self.instance.O)):
                    total = 0
                    for l in range(len(self.instance.L)):
                        total += self.z[l, n, m, o] * self.instance.kappa_l[l]
                    if total > self.instance.nu_nmo[n, m, o]:
                        return False
        # Constraint 5
        for n in range(len(self.instance.N)):
            total = 0
            for l in range(len(self.instance.L)):
                for m in range(len(self.instance.N)):
                    for o in range(len(self.instance.O)):
                        total += self.z[l, n, m, o] * self.instance.kappa_l[l]
            if total > self.instance.K_p_cap[n]:
                return False
        # Constraint 6
        for p in range(len(self.instance.N)):
            total = 0
            for l in range(len(self.instance.L)):
                for n in range(len(self.instance.N)):
                    for m in range(len(self.instance.N)):
                        for o in range(len(self.instance.O)):
                            total += (
                                self.z[l, n, m, o]
                                * self.instance.kappa_l[l]
                                * self.instance.d_nmop[n, m, o, p]
                            )
            if total > self.instance.K_cap[p]:
                return False
        return True

    # def get_valid_actions(self):
    #     """
    #     returns to the current assign idx position for optional actions（eg 0,1,...,max），pruning can be done according to constraints
    #     """
    #     # here is a simple return 0/1/2/3 can be adjusted according to the actual problem
    #     return [0, 1, 2, 3]

    def get_valid_actions(self):
        # """
        # returns to the current assign idx position for optional actions (eg 0,1,...,max),pruning according to constraints
        # """
        # # here is a simple return 0/1/2/3 can be adjusted according to the actual problem
        # return [0, 1, 2, 3]

        """
        returns to the current assign idx position for optional actions (eg 0,1,...,max),pruning according to constraints
        """
        idx = np.unravel_index(self.assign_idx, self.z.shape)
        l, n, m, o = idx

        # Constraint 1: sum_{m,o} z_{l,n,m,o} ≤ lambda_ln[l, n]
        remain1 = (
            self.instance.lambda_ln[l, n]
            - np.sum(self.z[l, n, :, :])
            + self.z[l, n, m, o]
        )

        # Constraint 2: sum_{o} z_{l,n,m,o} ≤ a_lnm[l, n, m] * x_lm[l, m] * lambda_ln[l, n]
        max2 = (
            self.instance.a_lnm[l, n, m]
            * self.instance.x_lm[l, m]
            * self.instance.lambda_ln[l, n]
        )
        remain2 = max2 - np.sum(self.z[l, n, m, :]) + self.z[l, n, m, o]

        # # Constraint 3: sum_{l} omega_l * sum_{n,o} z_{l,n,m,o} ≤ W_m[m]
        # total3 = 0
        # for ll in range(len(self.instance.L)):
        #     for nn in range(len(self.instance.N)):
        #         for oo in range(len(self.instance.O)):
        #             total3 += self.instance.omega_l[ll] * self.z[ll, nn, m, oo]
        # remain3 = self.instance.W_m[m] - total3 + self.instance.omega_l[l] * self.z[l, n, m, o]

        # # Constraint 4: sum_{l} z_{l,n,m,o} * kappa_l[l] ≤ nu_nmo[n, m, o]
        # total4 = 0
        # for ll in range(len(self.instance.L)):
        #     total4 += self.z[ll, n, m, o] * self.instance.kappa_l[ll]
        # remain4 = self.instance.nu_nmo[n, m, o] - total4 + self.instance.kappa_l[l] * self.z[l, n, m, o]

        # # Constraint 5: sum_{l} kappa_l[l] * sum_{m,o} z_{l,n,m,o} ≤ K_p_cap[n]
        # total5 = 0
        # for ll in range(len(self.instance.L)):
        #     for mm in range(len(self.instance.N)):
        #         for oo in range(len(self.instance.O)):
        #             total5 += self.z[ll, n, mm, oo] * self.instance.kappa_l[ll]
        # remain5 = self.instance.K_p_cap[n] - total5 + self.instance.kappa_l[l] * self.z[l, n, m, o]

        # # Constraint 6: sum_{l} kappa_l[l] * sum_{n,m,o} z_{l,n,m,o} * d_nmop[n, m, o, p] ≤ K_cap[p], ∀p
        # min_remain6 = float('inf')
        # for p in range(len(self.instance.N)):
        #     total6 = 0
        #     for ll in range(len(self.instance.L)):
        #         for nn in range(len(self.instance.N)):
        #             for mm in range(len(self.instance.N)):
        #                 for oo in range(len(self.instance.O)):
        #                     total6 += self.z[ll, nn, mm, oo] * self.instance.kappa_l[ll] * self.instance.d_nmop[nn, mm, oo, p]
        #     # current variable contribution to p
        #     contrib = self.instance.kappa_l[l] * self.instance.d_nmop[n, m, o, p] * self.z[l, n, m, o]
        #     remain6 = self.instance.K_cap[p] - total6 + contrib
        #     min_remain6 = min(min_remain6, remain6)

        # ### Consider all constraints
        # max_z = min(remain1, remain2, remain3, remain4, remain5, min_remain6)

        ### Only consider 2 constraints
        max_z = min(remain1, remain2)

        max_z = max(0, max_z)  # ensure non-negative

        ## wyb restrict the maximum action number, and later I consider handling the case where z exceeds n_actions in the trainer ## In fact, it cannot be handled, so it must be limited to less than n_action, because one hot encoding is used ## If onehot is not used, standard DQN/PPO implementation cannot be used directly
        # max_z = min(max_z, self.instance.n_action - 1)  # suppose n action has been passed in as an instance property
        max_z = min(
            10, max_z
        )  # the maximum action number is 10 which is the value of z so as to ensure that it does not exceed n action

        # assuming that the z maximum is not very large it can be enumerated directly
        return list(range(0, int(max_z) + 1))

    def get_assign_indices(self):
        """
        returns the l n m o quadruple subscript for the current assign idx under the linear expansion
        """
        return np.unravel_index(self.assign_idx, self.z.shape)
