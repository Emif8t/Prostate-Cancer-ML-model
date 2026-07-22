import pandas as pd


class Dataset:
    """
    Load and manage the dataset.
    """

    def __init__(self, filepath):

        self.filepath = filepath

        self.df = None

    def load(self):
        """
        Load dataset from an Excel file.
        """

        self.df = pd.read_excel(self.filepath)

        return self.df

    def split_target(self, target):
        """
        Split features and target variable.
        """

        X = self.df.drop(columns=[target])

        y = self.df[target]

        return X, y


    
       
