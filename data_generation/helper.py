from sklearn.preprocessing import LabelEncoder
from data_generation.compare_functions import *
from enums.diff_classifier_method_type import diff_classifier_method_type
import numpy as np
import pandas as pd
import _pickle as cPickle


# TODO: check if this function works for every dataframe not only for adult-dataset-df
def prepare_df(df, datasetname, class_name, replacing_mv=True):

    # set all columns of dataframe
    columns = df.columns.tolist()

    columns_for_decision_rules = list(columns)
    columns_for_decision_rules.remove(class_name)

    columns = columns[-1:] + columns[:-1]

    # 1. Step: set type for each feature/column of dataset
    type_features, features_type = recognize_features_type(df)

    continuous = type_features['integer'] + type_features['double']
    discrete = type_features['string']

    # discrete columns include always class_name column!
    # discrete, continuous = set_discrete_continuous(
    #     columns, type_features, class_name, discrete=None, continuous=None)

    # 2. Step: replace missing values of continuous and discrete features
    if replacing_mv:
        df = replace_by_median(df, continuous)
        df = replace_by_most_used_value(df, discrete)

    tmp_columns = list(columns)
    tmp_columns.remove(class_name)

    possible_outcomes = list(df[class_name].unique())

    # set index for features
    idx_features = {i: col for i, col in enumerate(
        list(tmp_columns))}

    # 3. Step: hot_encoding of all discrete features, except the to predicted column
    # df = pd.get_dummies(
    #    df, columns=[col for col in discrete if col != class_name])

    # 3. Step: label encoding of the discrete features
    df_le, label_encoder = label_encode(df, discrete)

    # 4. Step: extract features and class label
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    # 5. Step: preprare dataset dict
    dataset = {
        'name': datasetname,
        'df': df,
        'df_encoded': df_le,
        'columns': list(columns),
        'columns_for_decision_rules': list(columns_for_decision_rules),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset


# replace missing values of features with median of column
def replace_by_most_used_value(df, features):
    for col in features:
        most_used_value = df[col].value_counts().index[0]
        df[col].fillna(most_used_value, inplace=True)
    return df


# replace missing value of feature with most used value of column
def replace_by_median(df, features):
    for col in features:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    return df


# recognice different types (integer, double, string) of columns in dataframe.
def recognize_features_type(df):
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }

    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'

    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):

    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']

    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(
            set(continuous + type_features['integer'] + type_features['double']))

    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (
            f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))

    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder


def label_decode(df, columns, label_encoder, not_including_columns=[]):
    df_de = df.copy(deep=True)
    for col in columns:
        if col not in not_including_columns:
            le = label_encoder[col]
            df_de[col] = le.inverse_transform(df_de[col])
    return df_de


def dataframe2explain(X2E, dataset, idx_record2explain, blackbox):
    # Dataset to explit to perform explanation (typically is the train or test set (real instances))
    Z = cPickle.loads(cPickle.dumps(X2E))

    # Select record to predict and explain
    x = Z[idx_record2explain]

    # Remove record to explain (optional) from dataset Z and convert into dataframe
    # Z = np.delete(Z, idx_record2explain, axis=0)
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, x


def build_df2explain(bb, X, dataset):

    columns = dataset['columns']
    features_type = dataset['features_type']
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']

    y = bb.predict(X)
    yX = np.concatenate((y.reshape(-1, 1), X), axis=1)
    data = list()
    for i, col in enumerate(columns):
        data_col = yX[:, i]
        data_col = data_col.astype(int) if col in discrete else data_col
        data_col = data_col.astype(
            int) if features_type[col] == 'integer' else data_col
        data.append(data_col)
    # data = map(list, map(None, *data))
    data = [[d[i] for d in data] for i in range(0, len(data[0]))]
    dfZ = pd.DataFrame(data=data, columns=columns)
    dfZ = label_decode(dfZ, discrete, label_encoder)
    return dfZ


def get_dicts_for_classification(diff_classifier_method):
    if diff_classifier_method == diff_classifier_method_type.binary_diff_classifier:
        keys = {tuple(["dataset_diff", "compare", "diff"])}
        dict_dataset_diff = {"dataset_diff": None}
        dict_compare_functions = {"compare": compare_not_equal}
    if diff_classifier_method == diff_classifier_method_type.multiclass_diff_classifier:
        keys = {
            tuple(["dataset_multiclass_diff", "compare_multiclass", "diff_multiclass"])}
        dict_dataset_diff = {"dataset_multiclass_diff": None}
        dict_compare_functions = {"compare_multiclass": compare_multiclass}
    elif diff_classifier_method == diff_classifier_method_type.split_diff_classifiers:
        keys = {tuple(["dataset_00_diff", "compare_00", "diff_00"]), tuple(["dataset_11_diff", "compare_11",  "diff_11"]), tuple(
            ["dataset_01_diff",  "compare_01", "diff_01"]), tuple(["dataset_10_diff",  "compare_10", "diff_10"])}
        dict_dataset_diff = {"dataset_00_diff": None, "dataset_11_diff": None,
                             "dataset_01_diff": None, "dataset_10_diff": None}
        dict_compare_functions = {"compare_00": compare_00, "compare_11": compare_11,
                                  "compare_01": compare_01, "compare_10": compare_10}

    return keys, dict_dataset_diff, dict_compare_functions
