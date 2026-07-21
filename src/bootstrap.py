"""
bootstrap.py

Bootstrap confidence interval estimation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import trange

from .config import RANDOM_STATE, BOOTSTRAP_ITERATIONS


class BootstrapCI:
    """
    Bootstrap confidence interval calculations.
    """

    @staticmethod
    def auc(
        y_true,
        y_prob,
        n_iterations=BOOTSTRAP_ITERATIONS
    ):
        """
        Compute bootstrap confidence interval for ROC AUC.

        Parameters
        ----------
        y_true : array-like
            True class labels.

        y_prob : array-like
            Predicted probabilities.

        n_iterations : int
            Number of bootstrap iterations.

        Returns
        -------
        tuple
            Lower CI, median AUC, upper CI.
        """

        rng = np.random.default_rng(RANDOM_STATE)

        n = len(y_true)

        aucs = []

        for _ in trange(
            n_iterations,
            desc="Bootstrap"
        ):

            idx = rng.choice(
                np.arange(n),
                size=n,
                replace=True
            )

            yb = y_true.iloc[idx]
            pb = y_prob[idx]

            if len(np.unique(yb)) < 2:
                continue

            aucs.append(
                roc_auc_score(
                    yb,
                    pb
                )
            )

        return tuple(
            np.percentile(
                aucs,
                [2.5, 50, 97.5]
            )
        )
