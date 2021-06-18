from enum import Enum


class diff_classifier_method_type(Enum):
    binary_diff_classifier = 1
    multiclass_diff_classifier = 2
    split_diff_classifiers = 3
