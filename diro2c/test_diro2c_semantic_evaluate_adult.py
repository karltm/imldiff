import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import _pickle as cPickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score
from sklearn import svm
import copy
import seaborn as sn

import rule_extractor
import diro2c
from data import preprocess_data
from data_generation.neighborhood_generation import neighbor_generator
from data_generation.global_data_generation import global_data_generator
from data_generation.helper import *
import evaluation
from data.preprocess_data import _preprocess_adult_dataset

from enums.dataset_type import dataset_type
from enums.diff_classifier_method_type import diff_classifier_method_type

from yellowbrick.model_selection import FeatureImportances


def main():

    dataset_name = 'adult.csv'
    path_data = './data/datasets/'
    dataset = _preprocess_adult_dataset(path_data, dataset_name)

    X1, y1 = dataset['X'], dataset['y']
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=0)

    blackbox1 = DecisionTreeClassifier(random_state=0)
    blackbox1.fit(X1_train, y1_train)

    # print(dataset['columns_for_decision_rules'])
    fn = dataset['columns_for_decision_rules']
    cn = dataset['possible_outcomes']

    # fig1 = plt.figure(figsize=(16, 16), dpi=600)
    # tree.plot_tree(blackbox1,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig1.savefig('images/semantic_evaluation/bb1_decision_tree.png')

    # viz = FeatureImportances(blackbox1)
    # viz.fit(X1_train, y1_train)
    # viz.show()

    # -----------------------------------------------
    # manipulate and train BB 2:

    #print(np.unique(X1[:, 11], return_counts=True))

    X2 = copy.deepcopy(X1)
    y2 = copy.deepcopy(y1)
    for x, y in zip(X2, y2):
        # if y == 1:
        #x[8] = 10
        #x[0] = x[0] + 2
        x[7] = 1

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0)

    blackbox2 = DecisionTreeClassifier(random_state=0)
    blackbox2.fit(X2_train, y2_train)

    # fig2 = plt.figure(figsize=(16, 16), dpi=600)
    # tree.plot_tree(blackbox2,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig2.savefig('images/semantic_evaluation/bb2_decision_tree.png')

    # covMatrix = np.corrcoef(dataset['df_encoded'].to_numpy(), bias=False)
    # sn.heatmap(np.around(covMatrix, 2), annot=True, fmt='g')
    # plt.show()

    idx = 20
    print(X1_test[idx])
    X_to_recognize_diff = np.concatenate((X1_test, X2_test))
    diff_classifiers_info = diro2c.recognize_diff(idx, X_to_recognize_diff, dataset, blackbox1, blackbox2,
                                                  diff_classifier_method_type.binary_diff_classifier, data_generation_function=neighbor_generator.get_modified_genetic_neighborhood)

    # %matplotlib inline
    # gs = gridspec.GridSpec(3, 2)
    # fig = plt.figure(figsize=(14, 10))
    # labels = ['Logistic Regression', 'Decision Tree',
    #          'Random Forest', 'SVM', 'Naive Bayes', 'Neural Network']

    dc_info = diff_classifiers_info['binary_diff_classifer']
    dc_full = dc_info['dc_full']
    dc_test = dc_info['dc_test']
    evaluation_info = dc_info['evaluation_info']

    X_diff = evaluation_info['X']
    y_diff = evaluation_info['y']

    diffs = 0
    bb1_hit = 0
    bb2_hit = 0
    for xx, yy in zip(X_diff, y_diff):
        if yy == 1:
            if blackbox1.predict(np.array(xx).reshape(1, -1)) == 1:
                bb1_hit += 1
            diffs += 1

    print('diffs: ', diffs)
    print('bb1_hit: ', bb1_hit)
    if diffs != 0:
        print('hit_ratio: ', bb1_hit / diffs)
    else:
        print('hit_ratio: ', 0)

    y_test_true = evaluation_info['y_test_true']
    y_test_dc = evaluation_info['y_test_dc']

    print(f1_score(y_test_true, y_test_dc))

    print(np.unique(y_diff, return_counts=True))

    cn = ['no_diff', 'diff']
    # cn = ['00', '11', '10', '01']
    fig3 = plt.figure(figsize=(16, 16), dpi=600)
    tree.plot_tree(dc_full,
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig3.savefig('images/semantic_evaluation/dc_decision_tree.png')

    # rule_extractor.print_rules_for_binary(
    #     dc_full, dataset['columns_for_decision_rules'], ['no_diff', 'diff'], 'diff')


if __name__ == "__main__":
    main()
