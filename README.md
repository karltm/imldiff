# MOdel Comparison with Clustered difference clAssifier SHAP values (Mocca-SHAP)

This repository hosts the accompanying examples of my master thesis "Explaining the Differences of Decision Boundaries in Trained Classifiers"
which demonstrate how to apply the proposed model comparison method Mocca-SHAP.
It supports comparison of two classifiers and its explanations are based on the interpretability method [SHAP](https://shap.readthedocs.io/en/latest/).
The classifiers need to have a scikit-learn like interface.

The examples are in the form of jupyter notebooks.
You can start your own jupyter server or view them directly in your browser via [nbviewer](https://nbviewer.jupyter.org/github/MasterKarl/mocca-shap/tree/main/notebooks/).

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

Afterwards, run all commands in this environment in the root folder of the checked out repository. To deactivate, run `deactivate`.

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
python -m unittest test_difference_models test_shap test_explainers
```

## References

This approach is based on SHAP values, proposed in S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. In _Advances in Neural Information Processing Systems_, pages 4765–4774, 2017
The idea of the difference classifier was first published by Staufer and Rauber with [DIRO2C](https://gitlab.com/andsta/diro2c),
released under the GNU General Public License v3.0.
A copy has been obtained on 18th June 2021 from this [revision](https://gitlab.com/andsta/diro2c/-/commit/176095eba8740cac81cfbb9a545300018c8af82c). 
