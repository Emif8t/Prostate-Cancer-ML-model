class Preprocessor:

    @staticmethod
    def encode_group(df):

        df["Group"] = (

            df["Group"]

            .str.strip()

            .str.lower()

            .map({
                "control":0,
                "case":1
            })

        )

        return df


data = Dataset(path)

df = data.load()

df = Preprocessor.encode_group(df)
