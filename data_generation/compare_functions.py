import numpy as np
import warnings


def compare_not_equal(y1, y2):
    # compare function --> 0 (negative, no diff): same, 1 (positive): different predictions
    return np.invert(np.equal(y1, y2)).astype(int)


def compare_multiclass(y1, y2):
    y_diff = ([])

    for y1i, y2i in zip(y1, y2):
        if y1i == 0 and y2i == 0:
            y_diff.append(1)
        elif y1i == 1 and y2i == 1:
            y_diff.append(2)
        elif y1i == 1 and y2i == 0:
            y_diff.append(3)
        elif y1i == 0 and y2i == 1:
            y_diff.append(4)
        else:
            warnings.warn(
                'compare_multiclass: combination not defined: y1 & y2: ', y1i, y2i)

    return np.asarray(y_diff)


def compare_00(y1, y2):
    y_diff = ([])

    for y1i, y2i in zip(y1, y2):
        if y1i == 0 and y2i == 0:
            # combination detetected --> positive
            y_diff.append(1)
        else:
            y_diff.append(0)

    return np.asarray(y_diff)


def compare_11(y1, y2):
    y_diff = ([])

    for y1i, y2i in zip(y1, y2):
        if y1i == 1 and y2i == 1:
            # combination detetected --> positive
            y_diff.append(1)
        else:
            y_diff.append(0)

    return np.asarray(y_diff)


def compare_01(y1, y2):
    y_diff = ([])

    for y1i, y2i in zip(y1, y2):
        if y1i == 0 and y2i == 1:
            # combination detetected --> positive
            y_diff.append(1)
        else:
            y_diff.append(0)

    return np.asarray(y_diff)


def compare_10(y1, y2):
    y_diff = ([])

    for y1i, y2i in zip(y1, y2):
        if y1i == 1 and y2i == 0:
            # combination detetected --> positive
            y_diff.append(1)
        else:
            y_diff.append(0)

    return np.asarray(y_diff)
