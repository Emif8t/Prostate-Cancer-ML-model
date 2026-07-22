"""
visualization.py

Visualization utilities for machine learning model evaluation.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    CalibrationDisplay
)


class Visualizer:
    """
    Visualization utilities for model evaluation.
    """

    @staticmethod
    def plot_roc(results, title="ROC Curve"):
        """
        Plot ROC curves for all models.

        Parameters
        ----------
        results : dict
            Dictionary returned by NestedCrossValidator.

        title : str
            Plot title.
        """

        plt.figure(figsize=(8, 6))

        for model_name, result in results.items():

            RocCurveDisplay.from_predictions(

                result["Sample_Predictions"]["True_Label"],

                result["y_prob"],

                name=f"{model_name} (AUC={result['AUC']:.3f})"

            )

        plt.title(title)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_calibration(results, title="Calibration Curve"):
        """
        Plot calibration curves.
        """

        plt.figure(figsize=(8, 6))

        for model_name, result in results.items():

            CalibrationDisplay.from_predictions(

                result["Sample_Predictions"]["True_Label"],

                result["y_prob"],

                name=model_name

            )

        plt.title(title)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.show()
