"""
metrics.py

Utility functions for evaluating binary classification models.

This module computes common classification performance metrics
used for machine learning model evaluation.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

from .config import PROBABILITY_THRESHOLD


class Metrics:
    """
    Utility class for calculating binary classification metrics.
    """

    @staticmethod
    def calculate(
        y_true,
        y_pred=None,
        y_prob=None,
        threshold=PROBABILITY_THRESHOLD
    ):
        """
        Calculate binary classification performance metrics.

        Parameters
        ----------
        y_true : array-like
            True class labels.

        y_pred : array-like, optional
            Predicted class labels.

        y_prob : array-like, optional
            Predicted probabilities.

        threshold : float, default=0.5
            Probability threshold used to generate predicted classes
            when y_pred is not supplied.

        Returns
        -------
        dict
            Dictionary containing classification metrics.
        """

        # ----------------------------------------
        # Generate predicted classes if needed
        # ----------------------------------------

        if y_pred is None:

            if y_prob is None:
                raise ValueError(
                    "Either y_pred or y_prob must be provided."
                )

            y_pred = (np.asarray(y_prob) >= threshold).astype(int)

        # ----------------------------------------
        # Confusion matrix
        # ----------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        # ----------------------------------------
        # Sensitivity
        # ----------------------------------------

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        # ----------------------------------------
        # Specificity
        # ----------------------------------------

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        # ----------------------------------------
        # Results
        # ----------------------------------------

        return {

            "Accuracy": accuracy_score(
                y_true,
                y_pred
            ),

            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "F1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Brier": brier_score_loss(
                y_true,
                y_prob
            ),

            "Sensitivity": sensitivity,

            "Specificity": specificity

        }
