# Explaining Differences between Classifiers Using Interpretable Machine Learning

Here I host code and notebooks I'm using in my master's thesis to explain differences between machine learning classifiers using SHAP values. Mainly I'm using the python package [shap](https://github.com/slundberg/shap) and scikit-learn.

NOTE: It's still work-in-progress. Currently only the following notebooks make use of the most recent version of the difference classifiers, other notebooks use a slightly different approach that will be updated soon. Furthermore there are no notebooks yet to demonstrate the concept for non-binary base classifiers.

- [synthetic/2d_horizontally_separable/moved_decision_rule](https://github.com/MasterKarl/imldiff/tree/main/imldiff/notebooks/synthetic/2d_horizontally_separable/moved_decision_rule)

The structure of the notebooks for a specific task is always the same:
1. Create the data set and the base models, and checks the performance
2. Compare the predictions of the models using classical methods
3. Generate SHAP values for each of the base models and tries to compare them side-by-side or with subtraction.  I consider this the state-of-the art approach. There may be separate notebooks that make use of either the predicted labels only, the predicted probabilities or the log-odds (logit) of the predicted probabilities.
4. Generate SHAP values for the difference model(s) to explain the models' differences directly. Again, there may be several notebooks that use a different type of prediction.

## Usage

### Requirements
- Python 3.9 (other versions untested)
- Package `shap` requires llvm
  - on Mac OS, install with: `brew install llvm@12` and add to PATH variable
- Package `xgboost` requires LLVM's OpenMP runtime library (optional)
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
export PYTHONPATH=$PWD/imldiff
```

And start the jupyter server:
```
jupyter lab
```

### Run tests

```
python test.py
```

## Theory

Using interpretability methods, we can understand how a machine learning model behaves. By merging the output of two classifiers _A_ and _B_ and treating it as a special classification problem, we can also apply interpretability methods on that. The following approaches will be investigated at first, when the two classifiers _A_ and _B_ are restricted to binary classifiers:

1. Binary classification problem: `both classifiers agree` vs. `both classifiers disagree`
2. Three-class classification problem: `both classifiers agree`,  `A predicts the positive class and B the negative`, `B predicts positive and A negative`
3. Four-class classification problem: `A and B predict the positive class`, `A and B predict the negative class`, `A predicts the positive class and B the negative`, `B predicts positive and A negative`

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


## Feedback

I'd be glad to hear from your experiences with these tools or just any thoughts on it, especially if you're using it for university assignments 🙂 Either directly on github, or by mail to e1426356@student.tuwien.ac.at