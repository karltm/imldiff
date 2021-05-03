# Explaining Differences between Classifiers Using Interpretable Machine Learning

Here I host code and notebooks I'm using in my master's thesis to explain differences between machine learning classifiers using SHAP values. Mainly I'm using the python package [shap](https://github.com/slundberg/shap) and scikit-learn.

NOTE: It's still work-in-progress, and notebooks demonstrate the proposed approaches for binary classifiers only currently.

## Quickstart

### Requirements
- Python 3.9
- Package `shap` requires llvm version 8, 9 or 10
  - on Mac, install with: `brew install llvm@9 && echo 'export PATH="/usr/local/opt/llvm@9/bin:$PATH"' >> ~/.zshrc`
- Package `xgboost` requires LLVM's OpenMP runtime library (optional)
  - on Mac, install with: `brew install libomp`

### Install
TODO: instructions for setting it up in venv
```
pip install -r requirements.txt
```

### Run notebook server
On unix-based systems, run in project's root directory:
```
./start.sh
```

On windows, set `PYTHONPATH` to the imldiff directory before starting jupyter server.

## Theory

Using interpretability methods, we can understand how a machine learning model behaves. By merging the output of two classifiers $A$ and $B$ and treating it as a special classification problem, we can also apply interpretability methods on that. The following approaches will be investigated at first, when the two classifiers $A$ and $B$ are restricted to binary classifiers:

1. Binary classification problem: `both classifiers agree` vs. `both classifiers disagree`
2. Three-class classification problem: `both classifiers agree`,  `A predicts the positive class and B the negative`, `B predicts positive and A negative`
3. Four-class classification problem: `A and B predict the positive class`, `A and B predict the negative class`, `A predicts the positive class and B the negative`, `B predicts positive and A negative`

SHAP-values can be generated for the actually predicted labels, the predicted probabilities or log-odds if the classifiers support that. This enables a more detailled explanation.

Afterwards, I'll extend the approaches to explain differences between two multiclass classifiers.

## Implementation

SHAP values explain specific instances, but can be aggregated or their entire distribution visualized. I'll use them to:

- Explain the importance of the features for the observed differences, by aggregating them (TODO see figure 1) or plotting the distribution in a scatter plot
- Explain the marginal effect of a feature for the observed differences using SHAP dependence plots, similar to the partial dependence plots proposed by Friedman (Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." _Annals of statistics_ (2001): 1189-1232.)
- Cluster instances with similar SHAP values


