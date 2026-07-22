class Preprocessor:
    """
    Data preprocessing utilities.
    """

    @staticmethod
    def encode_group(df):
        """
        Encode Group column.

        Control -> 0
        Case -> 1
        """

        df = df.copy()

        df["Group"] = (

            df["Group"]

            .str.strip()

            .str.lower()

            .map({

                "control": 0,

                "case": 1

            })

        )

        return df
