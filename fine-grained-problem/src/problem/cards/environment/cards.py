import networkx as nx
import random
import heapq
import numpy as np
import torch

import numpy as np
from src.util.yen import k_shortest_path


class CardsProblem:
    # def __init__(self, L, N, O,
    #              lambda_ln, a_lnm, x_lm, omega_l, W_m,
    #              kappa_l, nu_nmo, K_p, d_nmop, K_p_cap, K_cap):
    def __init__(
        self,
        L,
        N,
        O,
        lambda_ln,
        a_lnm,
        x_lm,
        omega_l,
        W_m,
        kappa_l,
        nu_nmo,
        d_nmop,
        K_p_cap,
        K_cap,
    ):
        """
        initialize an instance of the cards integer programming problem the parameters are multi dimensional arrays or lists and the specific meanings are described in the model
        """
        self.L = L  # a collection of tasks
        self.N = N  # a collection of nodes
        self.O = O  # a collection of options

        # 参数
        self.lambda_ln = lambda_ln  # shape: (|L|, |N|)
        self.a_lnm = a_lnm  # shape: (|L|, |N|, |N|)
        self.x_lm = x_lm  # shape: (|L|, |N|)
        self.omega_l = omega_l  # shape: (|L|)
        self.W_m = W_m  # shape: (|N|)
        self.kappa_l = kappa_l  # shape: (|L|)
        self.nu_nmo = nu_nmo  # shape: (|N|, |N|, |O|)
        # self.K_p = K_p              # shape: (|N|)  # wyb:looks useless
        self.d_nmop = d_nmop  # shape: (|N|, |N|, |O|, |N|)
        self.K_p_cap = K_p_cap  # shape: (|N|)
        self.K_cap = K_cap  # shape: (|N|)

        ### decision variable initialization optional
        # self.z = np.zeros((len(L), len(N), len(N), len(O)), dtype=int)

    @staticmethod
    def generate_random_instance(n_L, n_N, n_O, seed):
        """
        a random instance of the cards problem is generated
        """
        if seed != -1:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(None)

        L = list(range(n_L))
        N = list(range(n_N))
        O = list(range(n_O))

        # lambda_ln = rng.integers(1, 4, size=(n_L, n_N))
        # a_lnm = rng.integers(0, 2, size=(n_L, n_N, n_N))
        # x_lm = rng.integers(0, 2, size=(n_L, n_N))
        # omega_l = rng.integers(1, 5, size=(n_L,))
        # W_m = rng.integers(10, 20, size=(n_N,))
        # kappa_l = rng.integers(1, 4, size=(n_L,))
        # nu_nmo = rng.integers(5, 15, size=(n_N, n_N, n_O))
        # K_p = rng.integers(10, 30, size=(n_N,))
        # d_nmop = rng.integers(1, 5, size=(n_N, n_N, n_O, n_N))
        # K_p_cap = rng.integers(10, 30, size=(n_N,))
        # K_cap = rng.integers(10, 30, size=(n_N,))

        ## wyb change to my generate
        # lambda_ln: (n_L, n_N)
        lambda_ln = rng.integers(0, 10, size=(n_L, n_N))
        # a_lnm: (n_L, n_N, n_N)
        a_lnm = rng.integers(0, 2, size=(n_L, n_N, n_N))
        # x_lm: (n_L, n_N), 0/1，50%probability
        x_lm = rng.integers(0, 2, size=(n_L, n_N))
        # omega_l: (n_L,)
        omega_l = rng.integers(10, 100, size=(n_L,))
        # W_m: (n_N,)
        W_m = rng.integers(10, 100, size=(n_N,))
        # kappa_l: (n_L,)
        kappa_l = rng.integers(10, 100, size=(n_L,))
        # nu_nmo: (n_N, n_N, n_O)
        nu_nmo = rng.integers(10, 100, size=(n_N, n_N, n_O))
        # d_nmop: (n_N, n_N, n_O, n_N)

        # ### dnmop wyb direct random
        # d_nmop = rng.integers(0, 2, size=(n_N, n_N, n_O, n_N))

        ## wyb use yen
        # 1. randomly generate adjacency matrices
        d_linjie_mu = 100
        d_linjie_sigma = 20
        adj = np.zeros((n_N, n_N))
        for i in range(n_N):
            for j in range(n_N):
                if i == j:
                    adj[i, j] = 0
                else:
                    adj[i, j] = abs(rng.normal(d_linjie_mu, d_linjie_sigma))

        # 2. Floyd algorithm finds the shortest circuit for all point pairs
        dist = adj.copy()
        for k in range(n_N):
            for i in range(n_N):
                for j in range(n_N):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # 3. the yen algorithm is used to find the shortest circuit of o for each pair
        d_nmop = np.zeros((n_N, n_N, n_O, n_N), dtype=int)
        for s in range(n_N):
            for t in range(n_N):
                # yen the node number starts at 1
                k_paths = k_shortest_path(dist.tolist(), s + 1, t + 1, n_O)
                for k_idx, (path, _) in enumerate(k_paths):
                    for node in path:
                        # node also from 1
                        d_nmop[s, t, k_idx, node - 1] = 1

        # K_p_cap: (n_N,)
        K_p_cap = rng.integers(10, 100, size=(n_N,))
        # K_cap: (n_N,)
        K_cap = rng.integers(10, 100, size=(n_N,))

        # return CardsProblem(L, N, O, lambda_ln, a_lnm, x_lm, omega_l, W_m, kappa_l, nu_nmo, K_p, d_nmop, K_p_cap, K_cap)
        return CardsProblem(
            L,
            N,
            O,
            lambda_ln,
            a_lnm,
            x_lm,
            omega_l,
            W_m,
            kappa_l,
            nu_nmo,
            d_nmop,
            K_p_cap,
            K_cap,
        )

    @staticmethod
    def generate_dataset(size, n_L, n_N, n_O, seed=0):
        """
        generate a dataset of cards problem instances in batches
        """
        dataset = []
        for i in range(size):
            instance = CardsProblem.generate_random_instance(n_L, n_N, n_O, seed)
            dataset.append(instance)
            seed += 1
        return dataset

