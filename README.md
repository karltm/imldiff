# Explaining Differences between Classifiers Using Interpretable Machine Learning

## Requirements
- Python 3.9 environment
- Package `shap` requires llvm version 8, 9 or 10
  on Mac, install with: `brew install llvm@9 && echo 'export PATH="/usr/local/opt/llvm@9/bin:$PATH"' >> ~/.zshrc`
- Package `xgboost` requires LLVM's OpenMP runtime library
  on Mac, install with: `brew install libomp`

## Install
```
pip install -r requirements.txt
```

## Run notebook server
```
jupyter lab
```

