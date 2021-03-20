import numpy as np


class Result:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def complement(self):
        return Result(f'{self}\'', 1 - self.values)

    def clip(self, lower_limit):
        return Result(f'max({self}, {lower_limit})', np.clip(self.values, lower_limit))

    def absolute(self):
        return Result(f'abs({self})', np.absolute(self.values))

    def add(self, other):
        return Result(f'({self} + {other})', self.values + other.values)

    def multiply(self, other):
        return Result(f'({self} * {other})', self.values * other.values)

    def odds(self):
        return Result(f'odds({self})', self.values / (1 - self.values))

    def ratio(self, other):
        return Result(f'({self}/{other})', self.values / other.values)

    def log(self):
        return Result(f'log({self})', np.log(self.values))

    def __str__(self):
        return self.name

    def __repr__(self):
        return repr(self.values)


def complement(result):
    return result.complement()


def clip(result, lower_limit):
    return result.clip(lower_limit)


def absolute(result):
    return result.absolute()


def add(result, other):
    return result.add(other)


def multiply(result, other):
    return result.multiply(other)


def odds(result):
    return result.odds()


def ratio(result, other):
    return result.ratio(other)


def log(result):
    return result.log()


def calculate_probability_of_a_positive_and_b_negative(proba_a, proba_b):
    return multiply(proba_a, complement(proba_b))


def calculate_probability_of_a_negative_and_b_positive(proba_a, proba_b):
    return multiply(complement(proba_a), proba_b)


def calculate_probability_of_both_different(proba_a, proba_b):
    return add(
        calculate_probability_of_a_positive_and_b_negative(proba_a, proba_b),
        calculate_probability_of_a_negative_and_b_positive(proba_a, proba_b)
    )


def calculate_log_of_odds_ratio(proba_a, proba_b):
    return log(ratio(odds(proba_a), odds(proba_b)))


def calculate_log_of_odds_ratio_of_complements(proba_a, proba_b):
    return log(ratio(odds(complement(proba_a)), odds(complement(proba_b))))


merge_functions = [
    calculate_probability_of_both_different,
    calculate_probability_of_a_positive_and_b_negative,
    calculate_probability_of_a_negative_and_b_positive,
    #calculate_log_of_odds_ratio,
    #calculate_log_of_odds_ratio_of_complements
]
