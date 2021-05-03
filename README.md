# Explaining Differences between Classifiers Using Interpretable Machine Learning

## Requirements
- Python 3.9
- Package `shap` requires llvm version 8, 9 or 10
  on Mac, install with: `brew install llvm@9 && echo 'export PATH="/usr/local/opt/llvm@9/bin:$PATH"' >> ~/.zshrc`
- Package `xgboost` requires LLVM's OpenMP runtime library (optional)
  on Mac, install with: `brew install libomp`

## Install
```
pip install -r requirements.txt
```

## Run notebook server
Before example notebooks can be run, you need to make the scripts in the project root available to the notebooks. On unix based systems, simply run `./start.sh` in the root directory, or set the `PYTHONPATH` variable manually before starting the jupyter notebook server.
```
./start.sh
```

