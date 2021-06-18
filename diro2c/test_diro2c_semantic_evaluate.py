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

from enums.dataset_type import dataset_type
from enums.diff_classifier_method_type import diff_classifier_method_type

from yellowbrick.model_selection import FeatureImportances


def main():

    # Create classification sample:
    X1, y1 = make_classification(
        n_samples=2000, n_features=10, scale=100, random_state=0)

    # -----------------------------------------------
    # train BB 1:

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=0)

    blackbox1 = DecisionTreeClassifier(random_state=0, max_depth=4)
    blackbox1.fit(X1_train, y1_train)

    fn = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
    cn = ['0', '1']
    fig1 = plt.figure(figsize=(16, 16), dpi=600)
    tree.plot_tree(blackbox1,
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig1.savefig('images/semantic_evaluation/bb1_decision_tree.png')

    # viz = FeatureImportances(blackbox1)
    # viz.fit(X1_train, y1_train)
    # viz.show()

    # -----------------------------------------------
    # manipulate and train BB 2:
    X2 = copy.deepcopy(X1)
    y2 = copy.deepcopy(y1)
    for x, y in zip(X2, y2):
        # if y == 1:
        x[1] = x[1] + 100

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0)

    blackbox2 = DecisionTreeClassifier(random_state=0, max_depth=4)
    blackbox2.fit(X2_train, y2_train)

    fig2 = plt.figure(figsize=(16, 16), dpi=600)
    tree.plot_tree(blackbox2,
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig2.savefig('images/semantic_evaluation/bb2_decision_tree.png')
    # -------------------------------------------------

    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    feature6 = []
    feature7 = []
    feature8 = []
    feature9 = []
    feature10 = []
    for x in X1:
        feature1.append(x[0])
        feature2.append(x[1])
        feature3.append(x[2])
        feature4.append(x[3])
        feature5.append(x[4])
        feature6.append(x[5])
        feature7.append(x[6])
        feature8.append(x[7])
        feature9.append(x[8])
        feature10.append(x[9])

    # for x in X2:
    #     feature1.append(x[0])
    #     feature2.append(x[1])

    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)
    feature3 = np.asarray(feature3)
    feature4 = np.asarray(feature4)
    feature5 = np.asarray(feature5)
    feature6 = np.asarray(feature6)
    feature7 = np.asarray(feature7)
    feature8 = np.asarray(feature8)
    feature9 = np.asarray(feature9)
    feature10 = np.asarray(feature10)

    data = np.array([feature1, feature2, feature3, feature4,
                     feature5, feature6, feature7, feature8, feature9, feature10])

    # covMatrix = np.corrcoef(data, bias=False)
    # sn.heatmap(np.around(covMatrix, 2), annot=True, fmt='g')
    # plt.show()

    y1 = y1.astype(str)
    d = {'feature_1': feature1, 'feature_2': feature2, 'feature_3': feature3, 'feature_4': feature4, 'feature_5': feature5, 'feature_6': feature6,
         'feature_7': feature7, 'feature_8': feature8, 'feature_9': feature9, 'feature_10': feature10, 'y': y1}
    df = pd.DataFrame(d)

    dataset = prepare_df(df, 'test', 'y')
    idx = 20
    print(X1_test[idx])
    X_to_recognize_diff = np.concatenate((X1_test, X2_test))
    diff_classifiers_info = diro2c.recognize_diff(idx, X_to_recognize_diff, dataset, blackbox1, blackbox2,
                                                  diff_classifier_method_type.binary_diff_classifier, data_generation_function=global_data_generator.get_global_training_test_dataset)

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

    fn = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
    cn = ['no_diff', 'diff']
    #cn = ['00', '11', '10', '01']
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
