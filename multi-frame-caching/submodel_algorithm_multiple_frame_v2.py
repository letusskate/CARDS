"""
we can only calculate the LP problem of this frame.
Because even if we calculate all the frames in one LP, we can only calculate by the order of frames
change output dir name
"""

# %%
import sys
import scipy
import tqdm
import numpy as np
from scipy.optimize import linprog
import time
import os


# %%
# model define
f_num = 5
l_num = 10
n_num = 15
o_num = 10
m_num = n_num
local_time_now = time.time()
local_time = time.localtime(local_time_now)
local_time = time.strftime("%Y%m%d-%H_%M_%S", local_time)
output_dir_path = "output_data"
input_dir_path = "data_dir"
output_file_name = local_time + "-calculate_log-frames" + ".txt"
input_file_name = os.path.join("framedataExample", "allframeExample.txt")
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
input_file = os.path.join(input_dir_path, input_file_name)
output_file = os.path.join(output_dir_path, output_file_name)


# %%
# data input
with open(input_file, mode="r") as txt_file:
    data = txt_file.readlines()
    for i in range(8, 20):
        exec(data[i])
del data


# %%
# Initial data
def initial_data(f):
    global lmd, k, d, w
    time_now = time.time()

    """ opt """
    c_opt = []
    for i in range(l_num):
        for j in range(n_num):
            for m in range(m_num):
                for o in range(o_num):
                    c_opt.append(-lmd[f * l_num * n_num + i * n_num + j])
    c_opt = np.array(c_opt)

    """ A_ub """
    A_ub = []
    # y_lnmo <= 1
    for i in range(l_num):
        for j in range(n_num):
            this_constraint = [0] * l_num * n_num * m_num * o_num
            for m in range(m_num):
                for o in range(o_num):
                    this_constraint[
                        i * n_num * m_num * o_num + j * m_num * o_num + m * o_num + o
                    ] = 1
            A_ub.append(this_constraint)
    # x_lm * r_l <= R_m
    for m in range(m_num):
        this_constraint = [0] * l_num * n_num * m_num * o_num
        for i in range(l_num):
            pass
        A_ub.append(this_constraint)
    # lmd_ln * k_l * y_lnmo * d_nmop <= K_p
    for p in range(n_num):
        this_constraint = [0] * l_num * n_num * m_num * o_num
        for i in range(l_num):
            for j in range(n_num):
                for m in range(m_num):
                    for o in range(o_num):
                        this_constraint[
                            i * n_num * m_num * o_num
                            + j * m_num * o_num
                            + m * o_num
                            + o
                        ] = (
                            lmd[f * l_num * n_num + i * n_num + j]
                            * k[i]
                            * d[
                                j * m_num * o_num * n_num
                                + m * o_num * n_num
                                + o * n_num
                                + p
                            ]
                        )
        A_ub.append(this_constraint)
    # w_l * lmd_ln * y_lnmo <= W_m
    for m in range(m_num):
        this_constraint = [0] * l_num * n_num * m_num * o_num
        for i in range(l_num):
            for j in range(n_num):
                for o in range(o_num):
                    this_constraint[
                        i * n_num * m_num * o_num + j * m_num * o_num + m * o_num + o
                    ] = (w[i] * lmd[f * l_num * n_num + i * n_num + j])
        A_ub.append(this_constraint)
    # y_lnmo <= a_lnm * x_lm
    for i in range(l_num):
        for j in range(n_num):
            for m in range(m_num):
                this_constraint = [0] * l_num * n_num * m_num * o_num
                for o in range(o_num):
                    this_constraint[
                        i * n_num * m_num * o_num + j * m_num * o_num + m * o_num + o
                    ] = 1
                A_ub.append(this_constraint)
    # x_ln * c_ln <= B
    this_constraint = [0] * l_num * n_num * m_num * o_num
    A_ub.append(this_constraint)
    A_ub = np.array(A_ub)

    """A_eq now useless """
    A_eq = None

    """ bounds of the var """
    bounds = []
    for i in range(l_num):
        for j in range(n_num):
            for m in range(m_num):
                for o in range(o_num):
                    bounds.append([0, None])
    bounds = np.array(bounds)

    """calculate time and return"""
    initial_time = time.time() - time_now
    print(f"initial time: {initial_time}")
    with open(file=output_file, mode="a+", encoding="utf-8") as txt_file:
        txt_file.write(f"initial time: {initial_time}\n")
    return c_opt, A_ub, A_eq, bounds


# %%
# LP solve function
def LP_solve(x_ln, last_ln, c_opt, A_ub, A_eq, bounds):
    """get data"""
    time_now = time.time()

    """ b_ub """
    b_ub = []
    # y_lnmo <= 1
    for i in range(l_num):
        for j in range(n_num):
            b_ub.append(1)
    # x_lm * r_l <= R_m
    for m in range(m_num):
        this_sum = 0
        for i in range(l_num):
            this_sum += x_ln[i * m_num + m] * r[i]
        b_ub.append(R[m] - this_sum)
    # lmd_ln * k_l * y_lnmo * d_nmop <= K_p
    for p in range(n_num):
        b_ub.append(K[p])
    # w_l * lmd_ln * y_lnmo <= W_m
    for m in range(m_num):
        b_ub.append(W[m])
    # y_lnmo <= a_lnm * x_lm
    for i in range(l_num):
        for j in range(n_num):
            for m in range(m_num):
                this_sum = 0
                for o in range(o_num):
                    this_sum += (
                        a[i * n_num * m_num + j * m_num + m] * x_ln[i * m_num + m]
                    )
                b_ub.append(this_sum)
    # x_ln * c_ln <= B
    this_sum = 0
    for i in range(l_num):
        for j in range(n_num):
            min_this_sum = 99999
            for jj in range(n_num):
                if last_ln[i * n_num + jj] == 0:
                    min_this_sum = min(
                        min_this_sum, x_ln[i * n_num + j] * c[i * n_num + j]
                    )
                else:
                    min_this_sum = min(
                        min_this_sum,
                        x_ln[i * n_num + j] * cc[i * n_num * n_num + jj * n_num + j],
                    )
            this_sum += min_this_sum
    b_ub.append(B[0] - this_sum)
    b_ub = np.array(b_ub)

    """ b_eq now useless """
    b_eq = None

    get_data_time = time.time() - time_now
    print(f"get data time: {get_data_time}", end="   ")

    """ calculate and return """
    time_now = time.time()
    # LINPROG_METHODS = ['simplex', 'revised simplex', 'interior-point', 'highs', 'highs-ds', 'highs-ipm']
    res = linprog(
        c=c_opt,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    calculate_time = time.time() - time_now
    print(f"calculate time: {calculate_time}")

    # print(res.status)
    if res.status == 2:
        return "NoSolution"
    min_value = res.fun
    # ylnm = res.x
    return min_value


# %%
# algorithm define
def generator():
    while True:
        yield


def greedy_algorithm():
    best_min_value_sum = 0
    last_ln = []
    for i in range(l_num * n_num):
        last_ln.append(0)
    for f in range(f_num):
        print((f"-----------------------frame {f+1}------------------------------"))
        with open(file=output_file, mode="a+", encoding="utf-8") as txt_file:
            txt_file.write(
                f"-----------------------frame {f+1}------------------------------\n"
            )
        c_opt, A_ub, A_eq, bounds = initial_data(f)
        best_min_value = 1e10
        x_ln_now = []
        for i in range(l_num * n_num):
            x_ln_now.append(0)
        round_cnt = 0
        # while True:
        for _ in tqdm.tqdm(generator()):  # make tqdm able to show the speed
            round_cnt += 1
            min_this_value = 1e10
            min_i = -1
            for i in range(l_num * n_num):
                time_now = time.time()
                if x_ln_now[i] == 1:
                    continue
                x_ln_this = x_ln_now.copy()
                x_ln_this[i] = 1
                this_value = LP_solve(
                    x_ln=x_ln_this,
                    last_ln=last_ln,
                    c_opt=c_opt,
                    A_ub=A_ub,
                    A_eq=A_eq,
                    bounds=bounds,
                )
                txt_file = open(file=output_file, mode="a+", encoding="utf-8")
                txt_file.write(
                    f"round:{round_cnt}, iter:{i+1}, this-value:{this_value}, time-cost:{time.time()-time_now}, "
                )
                if this_value == "NoSolution":
                    txt_file.write(
                        f"round-best-server:{min_i+1}, round-best-value:{min_this_value}\n"
                    )
                    txt_file.close()
                    continue
                elif this_value < min_this_value:
                    min_this_value = this_value
                    min_i = i
                txt_file.write(
                    f"round-best-server:{min_i+1}, round-best-value:{min_this_value}\n"
                )
                txt_file.close()
            if min_i != -1 and min_this_value < best_min_value:
                x_ln_now[min_i] = 1
                best_min_value = min_this_value
            else:
                with open(file=output_file, mode="a+", encoding="utf-8") as txt_file:
                    txt_file.write(
                        f"frame-final-object-value:{best_min_value}, frame-final-best-servers:{x_ln_now}\n"
                    )
                break
        best_min_value_sum += best_min_value
        last_ln = x_ln_now.copy()
    return best_min_value_sum


# %%
# run the algorithm
ans = greedy_algorithm()
print(ans)
