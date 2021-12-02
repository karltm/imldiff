from data_generation.neighborhood_generation.gpdatagenerator import calculate_feature_values
from data_generation.neighborhood_generation import neighbor_generator
from data_generation.helper import *
from data_generation.global_data_generation import global_data_generator
from enums.diff_classifier_method_type import diff_classifier_method_type

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from datetime import datetime

import _pickle as cPickle


def recognize_diff(idx_instance_to_rec_diff, X_to_recognize_diff, dataset, blackbox1, blackbox2, diff_classifier_method,
                   data_generation_function=neighbor_generator.get_modified_genetic_neighborhood,
                   discrete_use_probabilities=False, continuous_function_estimation=False, dtc_max_depth=None, gn_population_size=5000):
    datasetname = dataset['name']
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']

    # Dataset Preprocessing: calculate feature value spectrum
    dataset['feature_values'] = calculate_feature_values(X_to_recognize_diff, columns, class_name, discrete, continuous, 2000,
                                                         discrete_use_probabilities, continuous_function_estimation)

    # Deep copy of dataset (to recognize diffs) and fetch instance for detecting diff
    Z_to_rec_diff = cPickle.loads(cPickle.dumps(X_to_recognize_diff))
    x = Z_to_rec_diff[idx_instance_to_rec_diff]
    X_test_dataset = cPickle.loads(cPickle.dumps(dataset['X']))

    # Generate training/test dataset for diff-classifier(s)
    dict_dataset_diff = None

    # Genetic Neighborhood
    if data_generation_function is neighbor_generator.get_genetic_neighborhood:
        dict_dataset_diff = data_generation_function(
            x, blackbox1, blackbox2, dataset, Z_to_rec_diff, diff_classifier_method, population_size=gn_population_size)
    # Modified Genetic Neighborhood
    elif data_generation_function is neighbor_generator.get_modified_genetic_neighborhood:
        dict_dataset_diff = data_generation_function(
            x, blackbox1, blackbox2, dataset, Z_to_rec_diff, diff_classifier_method, population_size=gn_population_size)
    # Global Random Data and Modified Genetic Neighborhood
    elif data_generation_function is global_data_generator.get_global_mod_genetic_neighborhood_dataset:
        rd_size = int(len(X_to_recognize_diff) * 0.1)
        #rd_size = 10
        dict_dataset_diff = data_generation_function(blackbox1, blackbox2, dataset, X_to_recognize_diff, diff_classifier_method,
                                                     class_name, columns, discrete, continuous, features_type,
                                                     rd_size=rd_size, gn_population_size=gn_population_size, only_random=True)
    # Local (closed) Real Data
    elif data_generation_function is neighbor_generator.get_closed_real_data:
        dict_dataset_diff = data_generation_function(
            x, blackbox1, blackbox2, dataset, X_test_dataset, diff_classifier_method, k=2000)
    # Global Real Training/Testdata
    elif data_generation_function is global_data_generator.get_global_training_test_dataset:
        dataset_key = 'Dataset_' + datasetname + '_' + diff_classifier_method.name
        dict_dataset_diff = data_generation_function(
            dataset_key, blackbox1, blackbox2, dataset, X_test_dataset, diff_classifier_method)
    # Global Synthetic Random Data
    elif data_generation_function is global_data_generator.get_global_synthetic_random_dataset:
        dataset_key = 'Dataset_' + datasetname + '_' + diff_classifier_method.name
        dict_dataset_diff = data_generation_function(dataset_key, blackbox1, blackbox2, dataset, X_test_dataset, diff_classifier_method,
                                                     class_name, columns, discrete, continuous, features_type, size=1000, uniform=True)
        # for show_decision_boundaries.py changed size to 200:
        # dict_dataset_diff = data_generation_function(dataset_key, blackbox1, blackbox2, dataset, X_full_dataset, diff_classifier_method,
        #                                              class_name, columns, discrete, continuous, features_type, size=200, uniform=True)

    # TODO implement logic to train diff_classifier(s) and decide which classifier gets returned in case of split_diff_classifier_method
    # After consideration: return every 4 split_diff_classifiers to be able to differences in the dataset!
    diff_classifiers_info = {}

    if not dict_dataset_diff is None:

        y1_detect = blackbox1.predict(np.array(x).reshape(1, -1))[0]
        y2_detect = blackbox2.predict(np.array(x).reshape(1, -1))[0]

        for key in dict_dataset_diff:

            dataset = dict_dataset_diff[key]
            _, _, dict_compare_functions = get_dicts_for_classification(
                diff_classifier_method)

            if key == 'dataset_diff' and diff_classifier_method == diff_classifier_method_type.binary_diff_classifier:
                dc_name = 'binary_diff_classifer'
                compare_function = dict_compare_functions['compare']
            elif key == 'dataset_multiclass_diff' and diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
                dc_name = 'multiclass_diff_classifer'
                compare_function = dict_compare_functions['compare_multiclass']
            elif diff_classifier_method == diff_classifier_method_type.split_diff_classifiers:
                if key == 'dataset_00_diff':
                    dc_name = 'split_diff_classifier_00'
                    compare_function = dict_compare_functions['compare_00']
                if key == 'dataset_11_diff':
                    dc_name = 'split_diff_classifier_11'
                    compare_function = dict_compare_functions['compare_11']
                if key == 'dataset_01_diff':
                    dc_name = 'split_diff_classifier_01'
                    compare_function = dict_compare_functions['compare_01']
                if key == 'dataset_10_diff':
                    dc_name = 'split_diff_classifier_10'
                    compare_function = dict_compare_functions['compare_10']

            X = dataset['X']
            y = dataset['y']

            #print(np.unique(y, return_counts=True))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)

            # dc_test = RandomForestClassifier(n_estimators=20)
            # dc_test.fit(X_train, y_train)
            dc_test = DecisionTreeClassifier(max_depth=dtc_max_depth)
            dc_test.fit(X_train, y_train)

            dc_full = DecisionTreeClassifier(max_depth=dtc_max_depth)
            dc_full.fit(X, y)

            true_outcome = compare_function([y1_detect], [y2_detect])[0]
            dc_test_dataset_outcome = dc_test.predict(
                np.array(x).reshape(1, -1))[0]

            dc_full_outcome = dc_full.predict(
                np.array(x).reshape(1, -1))[0]

            y_test_dc = dc_test.predict(X_test)
            y_dc = dc_full.predict(X)

            y_test_dc_proba = dc_test.predict_proba(X_test)

            evaluation_info = {
                'dc_name': dc_name,
                'true_outcome': true_outcome,
                'dc_test_dataset_outcome': dc_test_dataset_outcome,
                'dc_full_outcome': dc_full_outcome,
                'X': X,
                'y': y,
                'X_test': X_test,
                'y_test_true': y_test,
                'y_test_dc': y_test_dc,
                'y_test_dc_proba': y_test_dc_proba,
                'y_true': y,
                'y_dc': y_dc,
                'df_diff': dataset['df']
            }

            # TODO return dict not idx array
            diff_classifiers_info[dc_name] = {
                'dc_full': dc_full,
                'dc_test': dc_test,
                'evaluation_info': evaluation_info
            }
    else:
        raise Exception('diff_classifier is None.')

    return diff_classifiers_info
