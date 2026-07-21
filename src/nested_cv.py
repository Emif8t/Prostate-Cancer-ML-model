"""
nested_cv.py

Nested cross-validation framework for supervised machine learning.

This module performs nested cross-validation, hyperparameter tuning,
final model training, and model evaluation using multiple machine
learning algorithms.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from .config import (
    RANDOM_STATE,
    OUTER_FOLDS,
    INNER_FOLDS
)

from .models import ModelFactory
from .pipeline import PipelineBuilder
from .bootstrap import BootstrapCI
from .metrics import Metrics


class NestedCrossValidator:
    """
    Perform nested cross-validation for multiple machine learning models.

    Parameters
    ----------
    model_factory : ModelFactory
        Factory containing machine learning models and
        their hyperparameter search spaces.
    """

    def __init__(self, model_factory: ModelFactory):

        self.model_factory = model_factory

        # Outer cross-validation
        self.outer_cv = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        # Inner cross-validation
        self.inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

    def run(self, X: pd.DataFrame, y: pd.Series):
        """
        Evaluate all machine learning models using nested
        cross-validation.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        y : pandas.Series
            Target labels.

        Returns
        -------
        tuple
            results : dict
                Performance metrics for all models.

            final_models : dict
                Optimized models trained on the full dataset.
        """

        results = {}

        final_models = {}

        models = self.model_factory.get_all_models()

        for model_name, model in models.items():

            print(f"\nTraining: {model_name}")

            result, final_model = self._evaluate_model(
                model_name=model_name,
                model=model,
                X=X,
                y=y
            )

            results[model_name] = result

            final_models[model_name] = final_model

        return results, final_models


from sklearn.model_selection import GridSearchCV

def _evaluate_model(
    self,
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series
):
    """
    Evaluate a single machine learning model using nested
    cross-validation.

    Parameters
    ----------
    model_name : str
        Name of the machine learning model.

    model : sklearn estimator
        Machine learning model.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target labels.

    Returns
    -------
    tuple
        Dictionary of model results and the final optimized model.
    """

    # -----------------------------------------
    # Storage
    # -----------------------------------------

    y_prob_all = np.zeros(len(y))

    sample_predictions = pd.DataFrame({

        "Sample_ID": X.index,

        "True_Label": y

    })

    # -----------------------------------------
    # Hyperparameter grid
    # -----------------------------------------

    param_grid = self.model_factory.get_param_grid(model_name)

    # -----------------------------------------
    # Outer Cross Validation
    # -----------------------------------------

    for train_idx, test_idx in self.outer_cv.split(X, y):

        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]

        # -----------------------------
        # Build pipeline
        # -----------------------------

        pipeline = PipelineBuilder.build(model)

        # -----------------------------
        # Hyperparameter tuning
        # -----------------------------

        grid = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grid,

            scoring="roc_auc",

            cv=self.inner_cv,

            n_jobs=-1

        )

        grid.fit(

            X_train,

            y_train

        )

        # -----------------------------
        # Best model
        # -----------------------------

        best_model = grid.best_estimator_

        y_prob = best_model.predict_proba(

            X_test

        )[:, 1]

        y_prob_all[test_idx] = y_prob

        sample_predictions.loc[

            test_idx,

            "Predicted_Probability"

        ] = y_prob

    # -----------------------------------------
    # Train final model on ALL samples
    # -----------------------------------------

    final_model, best_params = self._train_final_model(

        model_name=model_name,

        model=model,

        X=X,

        y=y

    )

    # -----------------------------------------
    # Calculate metrics
    # -----------------------------------------

    result = self._calculate_metrics(

        y=y,

        y_prob_all=y_prob_all,

        sample_predictions=sample_predictions,

        feature_names=X.columns.tolist(),

        best_params=best_params

    )

    return result, final_model
