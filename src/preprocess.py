import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = pd.read_excel(r"C:\Users\USER\Desktop\MLdata2.xlsx")

target_col = "Diagnosis"
df[target_col] = df[target_col].map({"BPH": 0, "Prostate": 1})

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)


Add preprocess.py 
