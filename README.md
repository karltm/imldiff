# Explaining Differences between Classifiers Using Interpretable Machine Learning

Here I host code and notebooks I'm using in my master's thesis to explain differences between machine learning classifiers using [SHAP values](https://shap.readthedocs.io/en/latest/).

Please see the demo notebooks for how to use the difference models. To visualize them directly in your browser, go to https://nbviewer.jupyter.org/github/MasterKarl/imldiff/tree/main/notebooks/.

## Usage

### Requirements
- Python 3.9
- llvm (required by shap package)
  - on Mac OS, install with: `brew install llvm@12` and add to PATH variable
- LLVM's OpenMP runtime library (required by xgboost package which is used in certain notebooks)
  - on Mac OS, install with: `brew install libomp`

### Install
It's easiest to install in a new virtual environment. Create one with your python 3.9 executable:

```
python -m venv .venv
```

Activate the virtual environment:
```
source .venv/bin/activate
```

And install the required packages:
```
pip install -r requirements.txt
```

Afterwards, run all commands in this environment. To deactivate, run `deactivate`.

### Run notebook server

First, set the PYTHONPATH environment variable, that the notebooks have access to the scripts:
```
export PYTHONPATH=$PWD
```

And start the jupyter server:
```
jupyter lab
```

### Run tests

```
python -m unittest
```

## Theory

Using interpretability methods, we can understand how a machine learning model behaves. By merging the output of two classifiers _A_ and _B_ and treating it as a special classification problem, we can also apply interpretability methods on that. The following approaches will be investigated at first, when the two classifiers _A_ and _B_ are restricted to binary classifiers:

1. Binary classification problem: `both classifiers agree` vs. `both classifiers disagree`
2. Four-class classification problem: `A and B predict the positive class`, `A and B predict the negative class`, `A predicts the positive class and B the negative`, `B predicts positive and A negative`

SHAP-values can be generated for, depending on the classifier:
- the actually predicted labels
- the predicted probabilities
- or log-odds

The latter two provide more pronounced results, because they take the uncertainty of the classifiers into account. Care should be taken whether probabilities or log-odds are explained if the classifiers support both, e.g. [logistic regression models are better explained using log-odds](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

Afterwards, I'll extend the approaches to explain differences between two multiclass classifiers.

## Implementation

SHAP values explain specific instances, but can be aggregated or their entire distribution visualized. I'll use them to:

- Explain the importance of the features for the observed differences, by aggregating them or plotting the distribution in a scatter plot
- Explain the marginal effect of a feature for the observed differences using SHAP dependence plots, similar to the partial dependence plots proposed by Friedman (Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." _Annals of statistics_ (2001): 1189-1232.)
- Cluster instances with similar SHAP values


## References

This approach is based on SHAP values, proposed in S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. In _Advances in Neural Information Processing Systems_, pages 4765–4774, 2017

The proposed approach is compared to [diro2c](https://gitlab.com/andsta/diro2c), released under the GNU General Public License v3.0. For this reason, a copy has been obtained on 18th June 2021 from this [revision](https://gitlab.com/andsta/diro2c/-/commit/176095eba8740cac81cfbb9a545300018c8af82c).
