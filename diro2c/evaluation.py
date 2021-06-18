from data_generation.neighborhood_generation import neighbor_generator
from data_generation.global_data_generation import global_data_generator

from enums.diff_classifier_method_type import diff_classifier_method_type

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import copy
import warnings


def evaluate_diff_classifier(x2detect, bb1, bb2, dc_test, dc_name, true_outcome, dc_test_dataset_outcome, dc_full_outcome,
                             X, y, diff_classifier_method, printing_metrics=False):
    y1_detect = bb1.predict(np.array(x2detect).reshape(1, -1))[0]
    y2_detect = bb2.predict(np.array(x2detect).reshape(1, -1))[0]

    # confusion_matrix = confusion_matrix(
    #     y_test_true, y_test_dc, labels=[0, 1])

    # hit_against_test_dataset = 1 if true_outcome == dc_test_dataset_outcome else 0
    # hit_against_full_dataset = 1 if true_outcome == dc_full_outcome else 0

    max_splits = min(np.unique(y, return_counts=True)[1])
    if max_splits >= 2:
        n_splits = 10
        if max_splits < 10:
            n_splits = max_splits

        warnings.filterwarnings('error')
        try:

            kfold = model_selection.StratifiedKFold(
                n_splits=n_splits, random_state=7, shuffle=True)

            rocauc_cv_scoring = 'roc_auc_ovo' if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier else 'roc_auc'
            cv_rocauc_results = model_selection.cross_val_score(
                dc_test, X, y, cv=kfold, scoring=rocauc_cv_scoring)

            f1_cv_scoring = 'f1_macro' if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier else 'f1'
            cv_f1_results = model_selection.cross_val_score(
                dc_test, X, y, cv=kfold, scoring=f1_cv_scoring)

            cv_acc_results = model_selection.cross_val_score(
                dc_test, X, y, cv=kfold, scoring='accuracy')

            cv_bal_acc_results = model_selection.cross_val_score(
                dc_test, X, y, cv=kfold, scoring='balanced_accuracy')

            if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
                cv_f1_micro_results = model_selection.cross_val_score(
                    dc_test, X, y, cv=kfold, scoring='f1_micro')

            # customized cross val calculation section:
            pearson_ccs = []
            mclass_recalls = []
            mclass_precisions = []
            for train_index, test_index in kfold.split(X, y):
                X_ind_train, X_ind_test = X[train_index], X[test_index]
                y_ind_train, y_ind_test = y[train_index], y[test_index]

                dc_ind_test = DecisionTreeClassifier().fit(X_ind_train, y_ind_train)
                y_ind_pred = dc_ind_test.predict(X_ind_test)

                pearson_cc_test = matthews_corrcoef(y_ind_test, y_ind_pred)

                pearson_ccs.append(pearson_cc_test)

                # multiclass individual recall and precision calculation:
                if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
                    pred_condition_positives = 0
                    condition_positives = 0
                    true_positives = 0
                    for x_r, y_r_true in zip(X_ind_test, y_ind_test):

                        y_diff_pred = dc_ind_test.predict(
                            np.array(x_r).reshape(1, -1))[0]

                        if y_r_true == 3 or y_r_true == 4:
                            condition_positives += 1

                        if y_diff_pred == 3 or y_diff_pred == 4:
                            pred_condition_positives += 1

                        if y_r_true == 3 and y_diff_pred == 3:
                            true_positives += 1
                        elif y_r_true == 4 and y_diff_pred == 4:
                            true_positives += 1

                    mclass_recall = true_positives / condition_positives
                    mclass_precision = true_positives / pred_condition_positives

                    mclass_recalls.append(mclass_recall)
                    mclass_precisions.append(mclass_precision)

            cv_pearson_cc_mean = np.mean(pearson_ccs)

            # recall & precision
            if diff_classifier_method == diff_classifier_method_type.binary_diff_classifier:
                cv_recall_results = model_selection.cross_val_score(
                    dc_test, X, y, cv=kfold, scoring='recall')

                cv_precision_results = model_selection.cross_val_score(
                    dc_test, X, y, cv=kfold, scoring='precision')

                cv_recall_mean = cv_recall_results.mean()
                cv_precision_mean = cv_precision_results.mean()
            elif diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
                cv_recall_mean = np.mean(mclass_recalls)
                cv_precision_mean = np.mean(mclass_precisions)

            if printing_metrics:
                print('dc_name: ', dc_name)
                print('y class distribution: ',
                      np.unique(y, return_counts=True))
                # print('y_test_true class distribution: ',
                #       np.unique(y_test_true, return_counts=True))
                # print('y_test_dc class distribution: ',
                #       np.unique(y_test_dc, return_counts=True))

                print('x2detect: ', x2detect)
                print('bb1: ', y1_detect)
                print('bb2: ', y2_detect)
                print('')
                print('true_outcome: ', true_outcome)
                print('dc_test_dataset_outcome : ', dc_test_dataset_outcome)
                print('')
                # print("confusion_matrix_test:")
                # print(confusion_matrix_test)
                print('--')
                print('%s: %f (%f)' %
                      ('cv_acc_results: ', cv_bal_acc_results.mean(), cv_bal_acc_results.std()))
                print('%s: %f (%f)' %
                      ('cv_f1_results: ', cv_f1_results.mean(), cv_f1_results.std()))
                print("matthews_corrcoef_test: ", cv_pearson_cc_mean)
                # print('roc_curve_test: ', roc_curve_test)

                # skplt.metrics.plot_roc(y_test_true, y_test_dc_proba)
                # plt.show()
                print('----------------------------------------------')

            if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
                evaluation_results = {
                    'cv_rocauc_results_mean': cv_rocauc_results.mean(),
                    'cv_f1_results_mean': cv_f1_results.mean(),
                    'cv_f1_micro_results_mean': cv_f1_micro_results.mean(),
                    'cv_acc_results_mean': cv_acc_results.mean(),
                    'cv_bal_acc_results_mean': cv_bal_acc_results.mean(),
                    'cv_pearson_cc_results_mean': cv_pearson_cc_mean,
                    'cv_recall_results_mean': cv_recall_mean,
                    'cv_precision_results_mean': cv_precision_mean,
                    'class_distribution': np.unique(y, return_counts=True)
                }
            else:
                evaluation_results = {
                    'cv_rocauc_results_mean': cv_rocauc_results.mean(),
                    'cv_f1_results_mean': cv_f1_results.mean(),
                    'cv_acc_results_mean': cv_acc_results.mean(),
                    'cv_bal_acc_results_mean': cv_bal_acc_results.mean(),
                    'cv_pearson_cc_results_mean': cv_pearson_cc_mean,
                    'cv_recall_results_mean': cv_recall_mean,
                    'cv_precision_results_mean': cv_precision_mean,
                    'class_distribution': np.unique(y, return_counts=True)
                }

        except Warning:
            evaluation_results = None
        except ZeroDivisionError:
            evaluation_results = None

    else:
        evaluation_results = None

    return max_splits, evaluation_results


# not up to date and not in use!
def evaluate_diff_split_classifiers(x2detect, bb1, bb2, diff_classifiers_info, diff_classifier_method):
    rocauc_res = []
    f1_res = []
    acc_res = []
    mcc_res = []
    max_splits = 0
    ret = None

    for key in diff_classifiers_info:
        diff_classifier_info = diff_classifiers_info[key]
        dc_full = diff_classifier_info[0]
        dc_test = diff_classifier_info[1]
        evaluation_info = diff_classifier_info[2]

        dc_name = evaluation_info['dc_name']

        true_outcome = evaluation_info['true_outcome']
        dc_test_dataset_outcome = evaluation_info['dc_test_dataset_outcome']
        dc_full_outcome = evaluation_info['dc_full_outcome']
        X = evaluation_info['X']
        y = evaluation_info['y']
        X_test = evaluation_info['X_test']
        y_test_true = evaluation_info['y_test_true']
        y_test_dc = evaluation_info['y_test_dc']
        y_test_dc_proba = evaluation_info['y_test_dc_proba']
        y_true = evaluation_info['y_true']
        y_dc = evaluation_info['y_dc']

        new_max_splits, evaluation_results = evaluate_diff_classifier(x2detect, bb1, bb2, dc_test, dc_name, true_outcome,
                                                                      dc_test_dataset_outcome, dc_full_outcome, X, y, diff_classifier_method)

        if not evaluation_results is None:

            if max_splits == 0 or new_max_splits < max_splits:
                max_splits = new_max_splits

            rocauc_res.append(evaluation_results['cv_rocauc_results_mean'])
            f1_res.append(evaluation_results['cv_f1_results_mean'])
            acc_res.append(evaluation_results['cv_acc_results_mean'])
            mcc_res.append(evaluation_results['cv_mcc_results_mean'])
        else:
            rocauc_res = None
            f1_res = None
            acc_res = None
            mcc_res = None
            break

    if not rocauc_res is None and not f1_res is None and not acc_res is None and not mcc_res is None:
        rocauc_results_mean = np.array(rocauc_res).mean()
        f1_results_mean = np.array(f1_res).mean()
        acc_results_mean = np.array(acc_res).mean()
        mcc_results_mean = np.array(mcc_res).mean()

        ret = {
            'cv_rocauc_results_mean': rocauc_results_mean,
            'cv_f1_results_mean': f1_results_mean,
            'cv_acc_results_mean': acc_results_mean,
            'cv_mcc_results_mean': mcc_results_mean
        }

    return max_splits, ret


def init_dict_evaluation_results_method(data_generation_functions, diff_classifier_methods):
    ret = {}
    for diff_classifier_method in diff_classifier_methods:
        dict_evaluation_results = {}
        for data_generation_function in data_generation_functions:
            d = dict_evaluation_results[get_keys_for_data_generation_function(
                data_generation_function)] = {}
            # d['cv_rocauc_results_mean'] = []
            # d['cv_f1_results_mean'] = []
            # d['cv_acc_results_mean'] = []
            # d['cv_mcc_results_mean'] = []
            # d['cv_recall_results_mean'] = []
            # d['cv_precision_results_mean'] = []
        ret[diff_classifier_method.name] = copy.deepcopy(
            dict_evaluation_results)
    return ret


def get_keys_for_data_generation_function(data_generation_function):
    if data_generation_function is neighbor_generator.get_genetic_neighborhood:
        return 'local_genetic_neighborhood'
    elif data_generation_function is neighbor_generator.get_closed_real_data:
        return 'local_real_data'
    elif data_generation_function is global_data_generator.get_global_synthetic_random_dataset:
        return 'global_synthetic_random_data'
    elif data_generation_function is global_data_generator.get_global_training_test_dataset:
        return 'global_real_data'
    elif data_generation_function is neighbor_generator.get_modified_genetic_neighborhood:
        return 'local_modified_genetic_neighborhood'
    else:
        warnings.warn('data_generation_function not implemented.')


def append_evaluation_result_grouped_by_method(dict_evaluation_results, evaluation_results, data_generation_function, diff_classifier_method):
    for key in evaluation_results:
        d = dict_evaluation_results[diff_classifier_method.name][get_keys_for_data_generation_function(
            data_generation_function)]
        if not key in d:
            d[key] = []
        d[key].append(copy.deepcopy(evaluation_results[key]))

    return dict_evaluation_results


def append_evaluation_result_grouped_by_metric(dict_evaluation_results, evaluation_results, data_generation_function, diff_classifier_method):
    for key in evaluation_results:
        if not key in dict_evaluation_results:
            dict_evaluation_results[key] = {}

        method_comb = diff_classifier_method.name + '_' + \
            get_keys_for_data_generation_function(data_generation_function)
        if not method_comb in dict_evaluation_results[key]:
            dict_evaluation_results[key][method_comb] = []
        dict_evaluation_results[key][method_comb].append(
            copy.deepcopy(evaluation_results[key]))
    return dict_evaluation_results
