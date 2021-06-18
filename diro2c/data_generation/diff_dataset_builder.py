from data_generation.compare_functions import *
from data_generation.helper import *

from enums.diff_classifier_method_type import diff_classifier_method_type

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def build_dict_dataset_diff(bb1, bb2, X, dataset, diff_classifier_method, x2detect=None, decoding=True):

    columns = dataset['columns']
    features_type = dataset['features_type']
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']

    keys, dict_dataset_diff, dict_compare_functions = get_dicts_for_classification(
        diff_classifier_method)

    for key in keys:

        # predicting for blackbox1 and blackbox2
        y1 = bb1.predict(X)
        y2 = bb2.predict(X)

        # np.savetxt("y1.csv",np.asarray(y1),delimiter=",")
        # np.savetxt("y2.csv",np.asarray(y2),delimiter=",")

        y = dict_compare_functions[key[1]](y1, y2)

        _testing_diff(X, y, x2detect, bb1, bb2, dict_compare_functions[key[1]])

        yX = np.concatenate((y.reshape(-1, 1), X), axis=1)
        #dict_Xy_Diff[key[1]] = [X, y]
        data = list()
        for i, col in enumerate(columns):
            data_col = yX[:, i]
            data_col = data_col.astype(int) if col in discrete else data_col
            data_col = data_col.astype(int) if features_type.get(
                col) == 'integer' else data_col
            data.append(data_col)
        # data = map(list, map(None, *data))
        data = [[d[i] for d in data] for i in range(0, len(data[0]))]

        new_class_name = key[2]
        columnsDiff = columns[1:]
        columnsDiff.insert(0, new_class_name)

        dfZ_Diff = pd.DataFrame(data=data, columns=columnsDiff)

        if decoding:
            dfZ_Diff = label_decode(
                dfZ_Diff, discrete, label_encoder, [dataset['class_name']])

            dataset_diff = prepare_df(
                dfZ_Diff, dataset['name'], new_class_name, replacing_mv=False)
            dict_dataset_diff[key[0]] = dataset_diff

        #dict_dfZ_Diff[key[2]] = dfZ_Diff

    # return dict_dataset, dict_Xy_Diff, dict_dfZ_Diff
    return dict_dataset_diff


def _testing_diff(X, y, x2detect, bb1, bb2, compare_function):
    # testing if diff-classifier would predict the same as bbs

    if not x2detect is None:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        diff_classifier = DecisionTreeClassifier()
        diff_classifier.fit(X_train, y_train)

        y_detect_diff = diff_classifier.predict(
            np.array(x2detect).reshape(1, -1))

        y1_detect = bb1.predict(np.array(x2detect).reshape(1, -1))
        y2_detect = bb2.predict(np.array(x2detect).reshape(1, -1))

        print("------------------")
        print("y1: ", y1)
        print("y2: ", y2)
        print("y_detect: ", compare_function(y1_detect, y2_detect))
        print("y_detect_diff: ", y_detect_diff)
        print("------------------")
