"""
nested_cv.py

Nested cross-validation framework for supervised machine learning.

This module performs nested cross-validation, hyperparameter tuning,
final model training, and model evaluation using multiple machine
learning algorithms.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

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
        Factory containing machine learning models and their
        corresponding hyperparameter search spaces.
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

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Evaluate all machine learning models using
        nested cross-validation.

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
                Dictionary containing performance metrics.

            final_models : dict
                Dictionary containing optimized models.
        """

        results = {}

        final_models = {}

        models = self.model_factory.get_all_models()

        for model_name, model in models.items():

            print("=" * 60)
            print(f"Training: {model_name}")
            print("=" * 60)

            result, final_model = self._evaluate_model(

                model_name=model_name,

                model=model,

                X=X,

                y=y

            )

            results[model_name] = result

            final_models[model_name] = final_model

        return results, final_models

    def _evaluate_model(
        self,
        model_name: str,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Evaluate a single machine learning model using
        nested cross-validation.

        Parameters
        ----------
        model_name : str
            Name of the machine learning model.

        model :
            Machine learning estimator.

        X : pandas.DataFrame
            Feature matrix.

        y : pandas.Series
            Target labels.

        Returns
        -------
        tuple
            Model results and optimized final model.
        """

        # -------------------------------------------------
        # Storage
        # -------------------------------------------------

        y_prob_all = np.zeros(len(y))

        sample_predictions = pd.DataFrame({

            "Sample_ID": X.index,

            "True_Label": y

        })

        # -------------------------------------------------
        # Hyperparameter grid
        # -------------------------------------------------

        param_grid = self.model_factory.get_param_grid(model_name)

        # -------------------------------------------------
        # Outer Cross Validation
        # -------------------------------------------------

        for train_idx, test_idx in self.outer_cv.split(X, y):

            X_train = X.iloc[train_idx]

            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]

            # Build pipeline
            pipeline = PipelineBuilder.build(model)

            # Hyperparameter optimization
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

            # Best model
            best_model = grid.best_estimator_

            y_prob = best_model.predict_proba(

                X_test

            )[:, 1]

            # Store predictions
            y_prob_all[test_idx] = y_prob

            sample_predictions.loc[
                test_idx,
                "Predicted_Probability"
            ] = y_prob

        # -------------------------------------------------
        # Train optimized model using the complete dataset
        # -------------------------------------------------

        final_model, best_params = self._train_final_model(

            model_name=model_name,

            model=model,

            X=X,

            y=y

        )

        # -------------------------------------------------
        # Calculate model performance
        # -------------------------------------------------

        result = self._calculate_metrics(

            y=y,

            y_prob_all=y_prob_all,

            sample_predictions=sample_predictions,

            feature_names=X.columns.tolist(),

            best_params=best_params

        )

        return result, final_model


#_train_final_model()
        def _train_final_model(
        self,
        model_name: str,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Train the optimized model on the complete dataset.

        Parameters
        ----------
        model_name : str
            Name of the machine learning model.

        model :
            Machine learning estimator.

        X : pandas.DataFrame
            Feature matrix.

        y : pandas.Series
            Target labels.

        Returns
        -------
        tuple
            final_model : fitted estimator

            best_params : dict
                Best hyperparameters selected by GridSearchCV.
        """

        pipeline = PipelineBuilder.build(model)

        param_grid = self.model_factory.get_param_grid(model_name)

        final_grid = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grid,

            scoring="roc_auc",

            cv=self.inner_cv,

            n_jobs=-1

        )

        final_grid.fit(X, y)

        final_model = final_grid.best_estimator_

        best_params = final_grid.best_params_

        return final_model, best_params


#_calculate_metrics()
        def _calculate_metrics(
        self,
        y: pd.Series,
        y_prob_all: np.ndarray,
        sample_predictions: pd.DataFrame,
        feature_names,
        best_params
    ):
        """
        Calculate model performance metrics.

        Parameters
        ----------
        y : pandas.Series
            True labels.

        y_prob_all : numpy.ndarray
            Predicted probabilities.

        sample_predictions : pandas.DataFrame
            Sample-level prediction table.

        feature_names : list
            Names of predictor variables.

        best_params : dict
            Best hyperparameters.

        Returns
        -------
        dict
            Dictionary containing all evaluation metrics.
        """

        # ------------------------------------------
        # Binary predictions
        # ------------------------------------------

        y_pred = (y_prob_all >= 0.5).astype(int)

        sample_predictions["Predicted_Class"] = y_pred

        sample_predictions["Correct"] = (

            sample_predictions["True_Label"]

            ==

            sample_predictions["Predicted_Class"]

        )

        # ------------------------------------------
        # Bootstrap AUC Confidence Interval
        # ------------------------------------------

        auc_low, auc, auc_high = BootstrapCI.auc(

            y,

            y_prob_all

        )

        # ------------------------------------------
        # Compute evaluation metrics
        # ------------------------------------------

        metrics = Metrics.calculate(

            y,

            y_pred,

            y_prob_all

        )

        # ------------------------------------------
        # Results dictionary
        # ------------------------------------------

        results = {

            # Predictions
            "y_prob": y_prob_all,

            "y_pred": y_pred,

            "Sample_Predictions": sample_predictions,

            # Metadata
            "feature_names": feature_names,

            "Best_Params": best_params,

            # ROC
            "AUC": auc,

            "AUC_Low": auc_low,

            "AUC_High": auc_high,

            # Classification metrics
            "Accuracy": metrics["Accuracy"],

            "Precision": metrics["Precision"],

            "Recall": metrics["Recall"],

            "F1": metrics["F1"],

            "Brier": metrics["Brier"],

            "Sensitivity": metrics["Sensitivity"],

            "Specificity": metrics["Specificity"]

        }

        return results
