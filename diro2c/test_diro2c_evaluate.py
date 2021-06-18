from data import preprocess_data
from data import manipulate_data
from data_generation.neighborhood_generation import neighbor_generator
from data_generation.global_data_generation import global_data_generator
from data_generation.helper import *

import evaluation
import diro2c

from enums.dataset_type import dataset_type
from enums.diff_classifier_method_type import diff_classifier_method_type

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import _pickle as cPickle
import numpy as np

import os
import datetime
import copy

import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import tree


def main():
    # TODO check random_state for reproducibility!!

    # datasets_to_test = [dataset_type.adult, dataset_type.bank_marketing,
    #                     dataset_type.credit_approval]

    datasets_to_test = [dataset_type.adult]

    #datasets_to_test = [dataset_type.bank_marketing]

    for dt in datasets_to_test:
        dataset = preprocess_data.getdataset(dt)

        X, y = dataset['X'], dataset['y']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        X2 = manipulate_data.manipulate_adult(X)

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y, test_size=0.2, random_state=0)

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=0)

        # X2 = manipulate_data.manipulate_bank_marketing(X)

        # X2_train, X2_test, y2_train, y2_test = train_test_split(
        #     X2, y, test_size=0.2, random_state=0)

        blackbox1 = DecisionTreeClassifier(random_state=0)
        blackbox1.fit(X_train, y_train)

        blackbox2 = DecisionTreeClassifier(random_state=0)
        blackbox2.fit(X2_train, y2_train)

        # fn = dataset['columns_for_decision_rules']
        # cn = ['0', '1']
        # fig, axes = plt.subplots(
        #     nrows=1, ncols=1, figsize=(4, 4), dpi=600)
        # tree.plot_tree(blackbox1,
        #                feature_names=fn,
        #                class_names=cn,
        #                filled=True)
        # fig.savefig('images/bb1_bank_marketing_decision_tree.png')

        # X_to_recognize_diff = np.unique(
        #    np.concatenate((X_test, X2_test)), axis=0)

        X_to_recognize_diff = np.concatenate((X_test, X2_test))
        #X_to_recognize_diff = X_test

        # data_generation_functions = [neighbor_generator.get_genetic_neighborhood,
        #                              neighbor_generator.get_closed_real_data,
        #                              global_data_generator.get_global_synthetic_random_dataset,
        #                              global_data_generator.get_global_training_test_dataset]

        data_generation_functions = [
            neighbor_generator.get_modified_genetic_neighborhood]

        # data_generation_functions = {
        #     global_data_generator.get_global_synthetic_random_dataset,
        #     neighbor_generator.get_closed_real_data}

        # diff_classifier_methods = [diff_classifier_method_type.binary_diff_classifier,
        #                            diff_classifier_method_type.multiclass_diff_classifier]

        diff_classifier_methods = [
            diff_classifier_method_type.binary_diff_classifier]

        # diff_classifier_methods = [diff_classifier_method_type.binary_diff_classifier,
        #                            diff_classifier_method_type.multiclass_diff_classifier,
        #                            diff_classifier_method_type.split_diff_classifiers]

        # X_to_recognize_diff = X_test
        test_runs = 10
        run = 1

        np.random.seed(0)
        chosen_indices = np.random.randint(
            0, len(X_test), size=test_runs)

        # indices only with data of original dataset
        # adult: [2732, 2607, 1653, 3264, 4931, 4859, 5827, 1033, 4373, 5874]
        # bank_marketing: [2732, 3264, 4859, 7891, 4373, 5874, 6744, 3468,  705, 2599]
        # credit_approval: [47, 117,  67, 103,   9,  21,  36,  87,  70,  88]

        dict_evaluation_results_grouped_by_method = evaluation.init_dict_evaluation_results_method(
            data_generation_functions, diff_classifier_methods)

        dict_evaluation_results_grouped_by_metric = {}

        for idx_record2explain in chosen_indices:

            Z = cPickle.loads(cPickle.dumps(X_to_recognize_diff))
            x = Z[idx_record2explain]

            for data_generation_function in data_generation_functions:
                for diff_classifier_method in diff_classifier_methods:

                    # diff_classifiers_info = diro2c.recognize_diff(idx_record2explain, X_to_recognize_diff, dataset, blackbox1, blackbox2,
                    #                                               diff_classifier_method, data_generation_function=data_generation_function,
                    #                                               dtc_max_depth=4)

                    diff_classifiers_info = diro2c.recognize_diff(idx_record2explain, X_to_recognize_diff, dataset, blackbox1, blackbox2,
                                                                  diff_classifier_method, data_generation_function=data_generation_function)

                    if diff_classifier_method == diff_classifier_method_type.binary_diff_classifier or diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:

                        dc_info = diff_classifiers_info['binary_diff_classifer' if diff_classifier_method ==
                                                        diff_classifier_method_type.binary_diff_classifier else 'multiclass_diff_classifer']
                        dc_full = dc_info['dc_full']
                        dc_test = dc_info['dc_test']

                        evaluation_info = dc_info['evaluation_info']

                        dc_name = evaluation_info['dc_name']
                        true_outcome = evaluation_info['true_outcome']
                        dc_test_dataset_outcome = evaluation_info['dc_test_dataset_outcome']
                        dc_full_outcome = evaluation_info['dc_full_outcome']
                        X = evaluation_info['X']
                        y = evaluation_info['y']
                        df_diff = evaluation_info['df_diff']

                        # X_test = evaluation_info['X_test']
                        # y_test_true = evaluation_info['y_test_true']
                        # y_test_dc = evaluation_info['y_test_dc']
                        # y_test_dc_proba = evaluation_info['y_test_dc_proba']
                        # y_true = evaluation_info['y_true']
                        # y_dc = evaluation_info['y_dc']

                        # max_splits, evaluation_results = evaluation.evaluate_diff_classifier(x, blackbox1, blackbox2, dc_test, dc_name, true_outcome,
                        #                                                                      dc_test_dataset_outcome, dc_full_outcome, X, y, X_test,
                        #                                                                      y_test_true, y_test_dc, y_test_dc_proba, y_true, y_dc, diff_classifier_method)

                        max_splits, evaluation_results = evaluation.evaluate_diff_classifier(x, blackbox1, blackbox2, dc_test, dc_name, true_outcome,
                                                                                             dc_test_dataset_outcome, dc_full_outcome, X, y, diff_classifier_method)

                        fn = dataset['columns_for_decision_rules']
                        cn = ['no_diff', 'diff']
                        fig, axes = plt.subplots(
                            nrows=1, ncols=1, figsize=(16, 8), dpi=600)
                        axes.autoscale()
                        tree.plot_tree(dc_full,
                                       feature_names=fn,
                                       class_names=cn,
                                       filled=True)
                        fig.savefig('images/decision_tree.png')

                        # fig, ax = plt.subplots(
                        #     nrows=1, ncols=1, figsize=(16, 16))

                        # fig = plot_decision_regions(X=X.astype(np.int64), y=y.astype(
                        #     np.int64), clf=dc_full,
                        #     feature_index=[8, 10],
                        #     filler_feature_values={0: np.average(X[:, 0]), 1: np.median(X[:, 1]), 2: np.median(X[:, 2]), 3: np.median(X[:, 3]),
                        #                            4: np.median(X[:, 4]), 5: np.median(X[:, 5]), 6: np.median(X[:, 6]), 7: np.median(X[:, 7]),
                        #                            9: np.average(X[:, 9]), 11: np.median(X[:, 11])},
                        #     filler_feature_ranges={
                        #         0: 10, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 9: 10, 11: 1},
                        #     legend=2, zoom_factor=0.6)
                        # ax[0].set_xlabel('feature 1')
                        # ax[0].set_ylabel('feature 2')

                        # plt.show()

                    else:
                        max_splits, evaluation_results = evaluation.evaluate_diff_split_classifiers(
                            x, blackbox1, blackbox2, diff_classifiers_info, diff_classifier_method)

                    # if max_splits >= 5:
                    if not evaluation_results is None:
                        dict_evaluation_results_grouped_by_method = evaluation.append_evaluation_result_grouped_by_method(
                            dict_evaluation_results_grouped_by_method, evaluation_results, data_generation_function, diff_classifier_method)

                        dict_evaluation_results_grouped_by_metric = evaluation.append_evaluation_result_grouped_by_metric(
                            dict_evaluation_results_grouped_by_metric, evaluation_results, data_generation_function, diff_classifier_method)

                        df_diff.to_csv('./test_reports/Diff_Dataset_' + bs + "_" + dt.name + "_" + diff_classifier_method.name + '_' +
                                       evaluation.get_keys_for_data_generation_function(data_generation_function) +
                                       '_Run' + str(run) + '.csv', index=False)

                    else:
                        df_diff.to_csv('./test_reports/Diff_Dataset_' + bs + "_" + dt.name + "_" + diff_classifier_method.name + '_' +
                                       evaluation.get_keys_for_data_generation_function(data_generation_function) +
                                       '_UNRATED_Run' + str(run) + '.csv', index=False)

            run += 1

        # ------------------------ print and store evaluation results ------------------------------

        now = datetime.datetime.now()
        smonth = '%02d' % now.month
        sday = '%02d' % now.day

        path = './test_reports/'

        # path = './test_reports/' + \
        #     str(now.year) + smonth + sday + '_' + str(now.hour) + '_Report/'
        # os.mkdir(path)

        tf = open(path + '_Report.txt', 'a')
        tf2 = open(path + '_Report.csv', 'a')

        print_and_write('Blackbox-Setting: ' + bs, tf)
        print_and_write('Dataset: ' + dataset['name'], tf)
        print_and_write('Evaluation Results:', tf)

        s = ''
        for index in chosen_indices:
            s = s + str(index) + ', '
        s = s[:-2]

        print_and_write('Test indices: ' + s, tf)
        print_and_write('', tf)
        print_and_write('Results grouped by diff-classifier:', tf)
        print_and_write("", tf)

        for method_key in dict_evaluation_results_grouped_by_method:
            print_and_write("   " + method_key + ": ", tf)
            for data_gen_key in dict_evaluation_results_grouped_by_method[method_key]:
                print_and_write("      " + data_gen_key + ": ", tf)
                print_and_write_csv(
                    bs + ";" + dataset['name'] + ";" + method_key + ";" + data_gen_key + ";", tf2)
                # print_and_write("         results :" +
                #                 dict_evaluation_results[method_key][data_gen_key], tf)

                for metric_key in dict_evaluation_results_grouped_by_method[method_key][data_gen_key]:

                    cv_metric_results_mean = dict_evaluation_results_grouped_by_method[
                        method_key][data_gen_key][metric_key]

                    if not metric_key == 'class_distribution':
                        print_and_write('         %s: %f (%f), min: %f, max: %f' %
                                        (metric_key, np.array(cv_metric_results_mean).mean(), np.array(cv_metric_results_mean).std(),
                                            np.array(cv_metric_results_mean).min(), np.array(cv_metric_results_mean).max()), tf)

                        print_and_write_csv('%f;%f;%f;%f;' %
                                            (np.array(cv_metric_results_mean).mean(), np.array(cv_metric_results_mean).std(),
                                                np.array(cv_metric_results_mean).min(), np.array(cv_metric_results_mean).max()), tf2)

                        if method_key == 'binary_diff_classifier' and metric_key == 'cv_f1_results_mean':
                            print_and_write_csv(';;;;', tf2)
                    else:
                        print_and_write(
                            '         ' + metric_key + ": ", tf)
                        print_and_write_modif(cv_metric_results_mean, tf)

                print_and_write_csv("\n", tf2)

        print_and_write(
            "---------------------------------------------", tf)
        print_and_write("Results grouped by metrics:", tf)
        print_and_write("", tf)

        for metric_key in dict_evaluation_results_grouped_by_metric:
            if not metric_key == 'class_distribution':
                print_and_write("   " + metric_key + ": ", tf)
                for method_comb in dict_evaluation_results_grouped_by_metric[metric_key]:
                    cv_metric_results_mean = dict_evaluation_results_grouped_by_metric[
                        metric_key][method_comb]

                    # TODO sort mean array
                    # sort.append(cv_metric_results_mean)

                    print_and_write('         %s: %f (%f)' %
                                    (method_comb, np.array(cv_metric_results_mean).mean(), np.array(cv_metric_results_mean).std()), tf)

        print_and_write(
            "=========================================================", tf)
        tf.close()
        tf2.close()


def print_and_write_csv(s, tf):
    tf.write(s)


def print_and_write(s, tf):
    print(s)
    tf.write(s + "\n")


def print_and_write_modif(s, tf):
    print("         ", s)
    tf.write("         ")
    tf.write(str(s))
    tf.write("\n")


if __name__ == "__main__":
    main()
