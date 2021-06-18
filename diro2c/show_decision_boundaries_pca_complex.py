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
from sklearn import svm
from sklearn.decomposition import PCA
import copy


import diro2c
from data import preprocess_data
from data import manipulate_data
from data_generation.neighborhood_generation import neighbor_generator
from data_generation.global_data_generation import global_data_generator
from data_generation.helper import *
import evaluation
import rule_extractor

from enums.dataset_type import dataset_type
from enums.diff_classifier_method_type import diff_classifier_method_type


def main():

    dataset = preprocess_data.getdataset(dataset_type.adult)

    # y_manipulated_1, y_manipulated_2 = manipulate_dataset_medianbased(
    #     dataset, dataset['X'], dataset['y'])

    y_manipulated = manipulate_data.manipulate_dataset_random(dataset['y'])

    pca = PCA(n_components=2)
    X = pca.fit_transform(dataset['X'])

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=0)

    blackbox1 = DecisionTreeClassifier()
    blackbox1.fit(X, dataset['y'])
    #blackbox1.fit(X, y_manipulated_1)

    blackbox2 = DecisionTreeClassifier()
    blackbox2.fit(
        X, y_manipulated)
    # blackbox2.fit(
    #     X, y_manipulated_2)

    feature1 = []
    feature2 = []
    for x in X:
        feature1.append(x[0])
        feature2.append(x[1])

    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)

    y = np.asarray(dataset['y']).astype(str)
    #y = np.asarray(y_manipulated_1).astype(str)

    d = {'y': y, 'x0': feature1, 'x1': feature2}

    df = pd.DataFrame(d)

    dataset = prepare_df(df, 'test', 'y')

    diff_classifiers_info = diro2c.recognize_diff(30, X, dataset, blackbox1, blackbox2,
                                                  diff_classifier_method_type.binary_diff_classifier, data_generation_function=neighbor_generator.get_genetic_neighborhood)

    dc_info = diff_classifiers_info['binary_diff_classifer']
    dc_full = dc_info['dc_full']
    dc_test = dc_info['dc_test']
    evaluation_info = dc_info['evaluation_info']

    X_diff = evaluation_info['X']
    y_diff = evaluation_info['y']

    print(np.unique(y_diff, return_counts=True))

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    fig = plot_decision_regions(X=X, y=np.asarray(dataset['y']).astype(
        np.int64), clf=blackbox1, ax=ax[0], legend=2)
    ax[0].set_xlabel('feature 1')
    ax[0].set_ylabel('feature 2')
    plt.title('blackbox 1')

    fig = plot_decision_regions(X=X, y=np.asarray(y_manipulated).astype(
        np.int64), clf=blackbox2, ax=ax[1], legend=2)
    ax[1].set_xlabel('feature 1')
    ax[1].set_ylabel('feature 2')
    plt.title('blackbox 2')

    fig = plot_decision_regions(X=X_diff, y=y_diff.astype(
        np.int64), clf=dc_full, ax=ax[2], legend=2)
    ax[2].set_xlabel('feature 1')
    ax[2].set_ylabel('feature 2')
    plt.title('binary diff-classifier')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles,
    #           ['class a', 'class b', 'class c', 'class d'],
    #           framealpha=0.3, scatterpoints=1)
    plt.show()
    plt.savefig('images/decision_boundaries_pca_komplex.png')

    fn = ['x - feature 1', 'y - feature 2']
    cn = ['negative', 'positive']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    tree.plot_tree(dc_full,
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig.savefig('images/decision_tree_pca_komplex.png')

    #tree_to_code(dc_full, dataset['columns'])
    #print_decision_tree(dc_full, dataset['columns'])
    rule_extractor.print_rules_for_binary(
        dc_full, dataset['columns_for_decision_rules'], ['no_diff', 'diff'], 'diff')


if __name__ == "__main__":
    main()
