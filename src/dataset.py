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
