"""
ADD flooding greedy to perform DOMF algorithm
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
f_num = 3  # at most 5
l_num = 5  # at most 10
n_num = 5  # at most 15
o_num = 5  # at most 10
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


def single_frame_greedy(f, last_ln):
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
    return best_min_value, x_ln_now


def expand_search(begin_f, x_ln_now):
    wait_for_search = []
    wait_for_search_frame = []
    for f in range(begin_f, f_num):
        wait_for_search.append([0] * (l_num * n_num))
        wait_for_search_frame.append(0)
    # first frame
    for i in range(l_num * n_num):
        if x_ln_now[0][i] == 1:
            wait_for_search_frame[0] = 1
            if len(wait_for_search) > 1:  # maybe flooding for only 1 frame
                wait_for_search[1][i] = 1
    # other frames
    for f in range(1, f_num - begin_f):
        for i in range(l_num * n_num):
            if x_ln_now[f][i] == 1:
                wait_for_search[f - 1][i] = 1
                wait_for_search_frame[f] = 1
                if f < f_num - begin_f - 1:
                    wait_for_search[f + 1][i] = 1
    # update wait_for_search
    for f in range(len(wait_for_search_frame)):
        if wait_for_search_frame[f] == 1:
            for i in range(l_num * n_num):
                wait_for_search[f][i] = 1
    return wait_for_search


def flooding_greedy(begin_f, last_f_ln):  # begin_f是可以取0的
    x_ln_now = []
    for f in range(begin_f, f_num):
        x_ln_now.append([0] * (l_num * n_num))
    wait_for_search = []
    for f in range(begin_f, f_num):
        wait_for_search.append([1] * (l_num * n_num))  # init the wait_for_search
    award_before = [0] * (
        f_num - begin_f
    )  # award for each frame also the final schedule obj

    # ### test wait_for_search
    # for f in range(begin_f,f_num):
    #     wait_for_search.append([0]*(l_num * n_num)) # init the wait_for_search
    # wait_for_search[1][6] = 1
    # wait_for_search[2][6] = 1

    round_cnt = 0
    for _ in tqdm.tqdm(generator()):  # make tqdm able to show the speed
        round_cnt += 1
        min_this_value = 1e10  # this round min of award
        min_this_value_change = 1e10  # this round min of award change
        min_i = -1  # this round min ln
        min_f = -1  # this round frame of ln
        for f in range(f_num - begin_f):
            if f == 0:
                last_ln = last_f_ln.copy()
            else:
                last_ln = x_ln_now[f - 1].copy()
            c_opt, A_ub, A_eq, bounds = all_initial_data[
                f + begin_f
            ]  # can be initialized only once outside the loop
            for i in range(l_num * n_num):
                if wait_for_search[f][i] == 0:
                    continue
                if x_ln_now[f][i] == 1:
                    continue
                time_now = time.time()
                x_ln_this = x_ln_now[f].copy()
                x_ln_this[i] = 1
                # ### test input
                # print('lastln:',last_ln,'x_ln_this:',x_ln_this,'c_opt:',c_opt,'A_ub:',A_ub,'A_eq:',A_eq,'bounds:',bounds)
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
                    f"FLOODING: round:{round_cnt}, frame:{begin_f+f+1}, iter:{i+1}, this-value:{this_value}, time-cost:{time.time()-time_now}, "
                )
                if this_value == "NoSolution":
                    txt_file.write(
                        f"||iter-of-best-cache:{min_i+1}, frame-of-best-cache:{min_f+begin_f+1}, round-best-value:{min_this_value}\n"
                    )
                    txt_file.close()
                    continue
                elif this_value - award_before[f] < min_this_value_change:
                    min_this_value = this_value
                    min_this_value_change = this_value - award_before[f]
                    min_i = i
                    min_f = f
                txt_file.write(
                    f"||iter-of-best-cache:{min_i+1}, frame-of-best-cache:{min_f+begin_f+1}, round-best-value:{min_this_value}\n"
                )
                txt_file.close()
        if min_i != -1 and min_this_value <= award_before[min_f]:
            x_ln_now[min_f][min_i] = 1
            award_before[min_f] = min_this_value
            wait_for_search = expand_search(
                begin_f, x_ln_now
            )  # expand the search space
            # ### test wait for search
            # print('wait_for_search update:',wait_for_search)
        else:
            with open(file=output_file, mode="a+", encoding="utf-8") as txt_file:
                txt_file.write(
                    f"FLOODING:all-frame-final-object-value:{award_before}, frame-final-best-servers:{x_ln_now}\n"
                )
            break
    return award_before, x_ln_now


def DOMF():
    # ### test expand search
    # x_ln_now = []
    # for f in range(f_num):
    #     x_ln_now.append([0] * (l_num * n_num))  # init the x_ln_now
    # x_ln_now[1][5] = 1
    # return expand_search(0, x_ln_now)  # begin_f=0, last_f_ln=[0]*(l_num * n_num) means the first frame has no last frame

    # ### test flooding greedy
    # return flooding_greedy(0, [0]*(l_num * n_num)) # begin_f=0, last_f_ln=[0]*(l_num * n_num) means the first frame has no last frame

    single_set = []
    flood_set = []

    print(f"initial single set and flood set: ")
    last_ln = [0] * (l_num * n_num)  # last frame has no last frame
    flooding_award, flooding_x_ln_now = flooding_greedy(0, last_ln)
    flood_set.append((flooding_award, flooding_x_ln_now))
    single_award, single_x_ln_now = [], []
    for i in range(f_num):
        awd, x_ln = single_frame_greedy(i, last_ln)  # get the best server of each frame
        single_award.append(awd)
        single_x_ln_now.append(x_ln)
    single_set.append((single_award, single_x_ln_now))

    print(f"begin from frame 2")
    for f in range(1, f_num):
        flood_new = []
        for flood in flood_set:
            single_award, single_x_ln_now = [], []
            single_last_ln = flood[1][f - 1]
            for i in range(f, f_num):
                awd, x_ln = single_frame_greedy(
                    i, single_last_ln
                )  # get the best server of each frame
                single_award.append(awd)
                single_x_ln_now.append(x_ln)
                single_last_ln = x_ln
            fnew = (flood[0][0:f] + single_award, flood[1][0:f] + single_x_ln_now)
            flood_new.append(fnew)
        single_new = []
        for single in single_set:
            flooding_award, flooding_x_ln_now = flooding_greedy(f, single[1][f - 1])
            snew = (single[0][0:f] + flooding_award, single[1][0:f] + flooding_x_ln_now)
            single_new.append(
                snew
            )  # get the best server of this frame based on the last frame
        for fnew in flood_new:
            single_set.append(fnew)
        for snew in single_new:
            flood_set.append(snew)
    # #### test output
    # print("!!! flood_set:", flood_set)
    # print("!!! single_set:", single_set)
    min_tuple = min(flood_set + single_set, key=lambda x: sum(x[0]))
    return min_tuple


# %%
# run the algorithm
all_initial_data = []
for i in range(f_num):
    all_initial_data.append(initial_data(i))

# ### test initial_data
# for i in range(f_num):
#     print('initial data opt:',all_initial_data[i][0])

ans = DOMF()
print(ans)
