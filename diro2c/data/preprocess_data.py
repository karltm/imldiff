from enums.dataset_type import dataset_type
from data_generation.helper import *

import pandas as pd


def getdataset(edataset):
    datapath = './data/datasets/'
    if edataset == dataset_type.adult:
        filename = "adult.csv"
        dataset = _preprocess_adult_dataset(datapath, filename)
    elif edataset == dataset_type.credit_approval:
        filename = "credit_approval.csv"
        dataset = _preprocess_credit_approval_dataset(datapath, filename)
    elif edataset == dataset_type.bank_marketing:
        filename = "bank-full.csv"
        dataset = _preprocess_bank_marketing_dataset(datapath, filename)
    else:
        raise Exception("dataset_type not implemented.")

    print("Dataset :", datapath + filename)

    return dataset


# private
# preprocessing of adult dataset
def _preprocess_adult_dataset(datapath, filename) -> 'Adult Dataset':

    # 1. Step: read adult csv and mark missing values
    missing_values = ['?']
    df = pd.read_csv(datapath + filename, delimiter=',',
                     skipinitialspace=True, na_values=missing_values)

    # and remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # preparing dataframe to dataset dict
    dataset = prepare_df(df, filename.replace('.csv', ''), 'income_class')

    return dataset


# private
# preprocessing of credit_approval dataset
def _preprocess_credit_approval_dataset(datapath, filename) -> 'Credit Approval Dataset':

    # 1. Step: read credit approval csv and mark missing values
    missing_values = ['?']
    df = pd.read_csv(datapath + filename, delimiter=',',
                     skipinitialspace=True, na_values=missing_values)

    # preparing dataframe to dataset dict
    dataset = prepare_df(df, filename.replace('.csv', ''), 'A16')

    return dataset


# private
# preprocessing of bank_marketing dataset
def _preprocess_bank_marketing_dataset(datapath, filename) -> 'Bank Marketing Dataset':

    # 1. Step: read bank marketing csv
    df = pd.read_csv(datapath + filename, delimiter=';', skipinitialspace=True)

    # preparing dataframe to dataset dict
    dataset = prepare_df(df, filename.replace('.csv', ''), 'y')

    return dataset
