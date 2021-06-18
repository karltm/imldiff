from data_generation.global_data_generation.global_data_store import GlobalDataStore
from data_generation.diff_dataset_builder import build_dict_dataset_diff
from data_generation.neighborhood_generation import neighbor_generator
from data_generation.neighborhood_generation.gpdatagenerator import calculate_feature_values

import numpy as np
import pandas as pd

import _pickle as cPickle


def get_global_training_test_dataset(dataset_key, blackbox1, blackbox2, dataset, X_real_dataset, diff_classifier_method):
    dict_natural_dataset_diff = GlobalDataStore.instance(
    ).load_dict_natural_dataset_diff(dataset_key)
    if dict_natural_dataset_diff is None:
        dict_natural_dataset_diff = build_dict_dataset_diff(
            blackbox1, blackbox2, X_real_dataset, dataset, diff_classifier_method)
        GlobalDataStore.instance().store_dict_natural_dataset_diff(
            dataset_key, dict_natural_dataset_diff)

    return dict_natural_dataset_diff


def get_global_synthetic_random_dataset(dataset_key, blackbox1, blackbox2, dataset, X, diff_classifier_method,
                                        class_name, columns, discrete, continuous, features_type, size=5000, uniform=True):
    dict_genetic_random_dataset_diff = GlobalDataStore.instance(
    ).load_dict_genetic_random_dataset_diff(dataset_key)

    if dict_genetic_random_dataset_diff is None:
        X1 = _generate_random_data(
            X, class_name, columns, discrete, continuous, features_type, size, uniform)

        dict_genetic_random_dataset_diff = build_dict_dataset_diff(
            blackbox1, blackbox2, X1, dataset, diff_classifier_method)

        GlobalDataStore.instance().store_dict_genetic_random_dataset_diff(
            dataset_key, dict_genetic_random_dataset_diff)

    return dict_genetic_random_dataset_diff


def get_global_mod_genetic_neighborhood_dataset(blackbox1, blackbox2, dataset, X_to_recognize_diff, diff_classifier_method,
                                                class_name, columns, discrete, continuous, features_type,
                                                rd_size=5000, uniform=True, gn_population_size=5000, only_random=True):
    X_uniform_random = _generate_random_data(
        X_to_recognize_diff, class_name, columns, discrete, continuous, features_type, rd_size, uniform, only_random=only_random)

    Z_to_rec_diff = cPickle.loads(cPickle.dumps(X_uniform_random))

    X_union = None
    y_union = None

    for x in Z_to_rec_diff:

        # dataset['feature_values'] = calculate_feature_values(Z_to_rec_diff, columns, class_name, discrete, continuous, 2000,
        #                                                      False, False)

        dict_dataset_diff = neighbor_generator.get_modified_genetic_neighborhood(
            x, blackbox1, blackbox2, dataset, Z_to_rec_diff, diff_classifier_method, population_size=gn_population_size)

        # TODO change for multiclass
        diff_dataset = dict_dataset_diff['dataset_diff']

        if X_union is None and y_union is None:
            X_union = diff_dataset['X']
            y_union = diff_dataset['y']
        else:
            X_union = np.concatenate((X_union, diff_dataset['X']))
            y_union = np.concatenate((y_union, diff_dataset['y']))

    dict_dataset_diff = build_dict_dataset_diff(
        blackbox1, blackbox2, X_union, dataset, diff_classifier_method)

    return dict_dataset_diff


def _generate_random_data(X, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True, only_random=False):
    X1 = list()
    columns1 = list(columns)
    columns1.remove(class_name)
    for i, col in enumerate(columns1):
        values = X[:, i]
        diff_values = np.unique(values)
        prob_values = [1.0 * list(values).count(val) / len(values)
                       for val in diff_values]
        if col in discrete:
            if uniform:
                new_values = np.random.choice(diff_values, size)
            else:
                new_values = np.random.choice(
                    diff_values, size, prob_values)
        elif col in continuous:
            mu = np.mean(values)
            sigma = np.std(values)
            if sigma <= 0.0:
                new_values = np.array([values[0]] * size)
            else:
                new_values = np.random.normal(mu, sigma, size)
        if features_type[col] == 'integer':
            new_values = new_values.astype(int)
        X1.append(new_values)

    if only_random:
        X1 = np.vstack((X1[0], X1[1])).T
    else:
        X1 = np.concatenate((X, np.column_stack(X1)), axis=0).tolist()
        X1 = np.unique(X1, axis=0)

    return X1
