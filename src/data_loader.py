import pandas as pd
import numpy as np

def load_data(r"C:\Users\USER\Desktop\New.xlsx"):
    df = pd.read_excel(r"C:\Users\USER\Desktop\New.xlsx")
    return df
    
    target_col = "Cancer_diagnosis"
    df[target_col] = df[target_col].map({"BPH": 0, "Prostate": 1})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

  
Add data_loader.py
