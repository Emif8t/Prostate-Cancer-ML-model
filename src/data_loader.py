import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """
    For now, we use a sample dataset from sklearn.
    Later, replace with SNP/prostate cancer dataset.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y
  
Add data_loader.py
