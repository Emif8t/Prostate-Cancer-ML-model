"""
exporter.py

Utilities for exporting machine learning results.
"""

import pandas as pd


class Exporter:
    """
    Export model outputs.
    """

    @staticmethod
    def sample_predictions(
        results,
        model_name,
        filename
    ):
        """
        Export sample-level predictions.
        """

        results[model_name][
            "Sample_Predictions"
        ].to_excel(

            filename,

            index=False

        )

    @staticmethod
    def metrics_table(
        results,
        filename
    ):
        """
        Export model performance metrics.
        """

        rows = []

        for model_name, result in results.items():

            rows.append({

                "Model": model_name,

                "AUC": result["AUC"],

                "Accuracy": result["Accuracy"],

                "Precision": result["Precision"],

                "Recall": result["Recall"],

                "F1": result["F1"],

                "Sensitivity": result["Sensitivity"],

                "Specificity": result["Specificity"],

                "Brier": result["Brier"]

            })

        pd.DataFrame(rows).to_excel(

            filename,

            index=False

        )
