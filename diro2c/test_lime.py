import lime
import lime.lime_tabular
from data_generation.helper import *

from data.preprocess_data import _preprocess_adult_dataset
from data.preprocess_data import _preprocess_bank_marketing_dataset
from data.preprocess_data import _preprocess_credit_approval_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def main():

    dataset_name = 'adult.csv'
    path_data = './data/datasets/'
    dataset = _preprocess_adult_dataset(path_data, dataset_name)

    # dataset_name = 'bank-full.csv'
    # path_data = './data/datasets/'
    # dataset = _preprocess_bank_marketing_dataset(path_data, dataset_name)

    # dataset_name = 'credit_approval.csv'
    # path_data = './data/datasets/'
    # dataset = _preprocess_credit_approval_dataset(path_data, dataset_name)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    blackbox = DecisionTreeClassifier(random_state=0)
    blackbox.fit(X_train, y_train)

    X_to_recognize_diff = X_test
    # idcs_record2explain = [2732, 2607, 1653,
    #                        3264, 4931, 4859, 5827, 1033, 4373, 5874]
    idcs_record2explain = [20]
    for idx_record2explain in idcs_record2explain:

        class_name = dataset['class_name']
        columns = dataset['columns']
        continuous = dataset['continuous']
        possible_outcomes = dataset['possible_outcomes']
        label_encoder = dataset['label_encoder']

        feature_names = list(columns)
        feature_names.remove(class_name)

        categorical_names = dict()
        idx_discrete_features = list()
        for idx, col in enumerate(feature_names):
            if col == class_name or col in continuous:
                continue
            idx_discrete_features.append(idx)
            categorical_names[idx] = label_encoder[col].classes_

        dfZ, x_2_exp = dataframe2explain(
            X_to_recognize_diff, dataset, idx_record2explain, blackbox)
        print("Instance: ")
        print(dfZ.iloc[[idx_record2explain]])

        print("")
        print("Feature importance:")

        # Create Lime Explanator
        num_features = 10
        explainer = lime.lime_tabular.LimeTabularExplainer(X_to_recognize_diff,
                                                           feature_names=feature_names,
                                                           class_names=possible_outcomes,
                                                           categorical_features=idx_discrete_features,
                                                           categorical_names=categorical_names,
                                                           verbose=False
                                                           )

        exp, Zlr, Z, lr = explainer.explain_instance(X_to_recognize_diff[idx_record2explain], blackbox.predict_proba,
                                                     num_features=num_features, num_samples=1000)

        used_features_idx = list()
        used_features_importance = list()
        logic_explanation = list()
        for idx, weight in exp.local_exp[1]:
            used_features_idx.append(idx)
            used_features_importance.append(weight)
            logic_explanation.append(
                exp.domain_mapper.discretized_feature_names[idx])

        for feature, weight in zip(logic_explanation, used_features_importance):
            print(feature, round(weight, 3))

        # print(X_to_recognize_diff[idx_record2explain])
        # bb_outcome = blackbox.predict(Z[0].reshape(1, -1))[0]

        # print(bb_outcome)
        # cc_outcome = np.round(lr.predict(Zlr[0, used_features_idx].reshape(1, -1))).astype(int)[0]
        #
        # y_pred_bb = blackbox.predict(Z)
        # y_pred_cc = np.round(lr.predict(Zlr[:, used_features_idx])).astype(int)


if __name__ == "__main__":
    main()
