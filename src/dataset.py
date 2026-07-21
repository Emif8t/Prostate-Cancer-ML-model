import pandas as pd


class Dataset:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        """
        data = Dataset("data/ROCdata2.xlsx")

        df = data.load()
        """
        self.df = pd.read_excel(self.filepath)

        return self.df

    def split_target(self, target):

        X = self.df.drop(columns=[target])

        y = self.df[target]

        return X, y

        X, y = data.split_target("Group")
