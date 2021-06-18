import numpy as np
import copy


def manipulate_adult(X):
    man_X = copy.deepcopy(X)
    for x0 in man_X:
        x0[8] = x0[8] + 7000

    return man_X


def manipulate_adult2(X):
    man_X = copy.deepcopy(X)
    for x0 in man_X:
        x0[0] -= 5
        x0[8] = x0[8] + 7000

    return man_X


def manipulate_credit_approval(X):
    man_X = copy.deepcopy(X)
    for x0 in man_X:
        x0[9] = (x0[9] + 1) % 2

    return man_X


def manipulate_bank_marketing(X):
    man_X = copy.deepcopy(X)
    for x0 in man_X:
        x0[13] = x0[13] + 4

    return man_X


# def manipulate_cont_attribute(X, y, col_X_idx, factor):
#     ret_X = []
#     for x0, y0 in zip(X, y):
#         if y0 == 0:
#             x0[col_X_idx] = x0[col_X_idx] / factor
#         else:
#             x0[col_X_idx] = x0[col_X_idx] * factor

#         ret_X.append(x0)

#     return ret_X, y


# def manipulate_adult_relationship_cgain(X, y, factor):
#     ret_X = []
#     for x0, y0 in zip(X, y):
#         if x0[8] == 0:
#             x0[8] = (-1 * factor)
#         else:
#             x0[8] = x0[8] * factor

#         x0[5] = (x0[5] + 1) % 6  # relationship has 6 possible values

#         ret_X.append(x0)

#     return ret_X, y

# def manipulate_adult_relationship_cgain(X, y, factor):
#     man_X = copy.deepcopy(X)
#     for x0, y0 in zip(man_X, y):
#         if y0 == 0:
#             x0[10] = x0[10] - factor
#             x0[8] = x0[8] - factor
#             x0[5] = (x0[5] + 1) % 6
#         else:
#             x0[10] = x0[10] + factor
#             x0[8] = x0[8] + factor
#             x0[5] = (x0[5] + 1) % 6

#     return man_X


# def manipulate_adult_relationship_cgain(X, y, factor):
#     man_X = copy.deepcopy(X)
#     for x0 in man_X:
#         x0[10] = x0[10] + factor
#         x0[8] = x0[8] + factor
#         x0[5] = (x0[5] + 1) % 6

#     return man_X, y


# def manipulate_credit_A9_A15(X, y, factor):
#     ret_X = []
#     for x0, y0 in zip(X, y):
#         if y0 == 0:
#             x0[14] = x0[14] / factor
#             x0[8] = (x0[8] + 1) % 2  # A9 has 2 possible values
#         else:
#             x0[14] = x0[14] * factor
#             x0[8] = (x0[8] + 1) % 2  # A9 has 2 possible values

#         ret_X.append(x0)

#     return ret_X, y

# def manipulate_credit_A9_A15(X, factor):
#     man_X = copy.deepcopy(X)
#     for x0 in man_X:
#         x0[14] = x0[14] / factor
#         x0[8] = (x0[8] + 1) % 2

#     return man_X


# def manipulate_bank_marketing_duration_pdays(X, y, factor):
#     ret_X = []
#     for x0, y0 in zip(X, y):
#         if y0 == 0:
#             x0[11] = x0[11] / factor
#             x0[13] = x0[13] / factor
#         else:
#             x0[11] = x0[11] * factor
#             x0[13] = x0[13] * factor

#         ret_X.append(x0)

#     return ret_X, y


# not in use
def manipulate_dataset_random(y):
    i = 1
    ret = []
    np.random.seed(9)
    y_rnd = np.random.randint(2, size=len(y))
    for y1, y1_rnd in zip(y, y_rnd):
        if (i % 2) == 0:
            ret.append(y1)
        else:
            ret.append(y1_rnd)
        i += 1
    return y_rnd


def manipulate_dataset_medianbased(idx2manipulate, X, y):
    median = np.median(X[idx2manipulate])
    ret1 = []
    ret2 = []

    for x, y in zip(X, y):
        if x[idx2manipulate] > median:
            ret1.append(1)
            ret2.append(0)
        else:
            ret1.append(y)
            ret2.append(y)

    return ret1, ret2
