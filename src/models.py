"""
models.py

Machine learning model definitions and hyperparameter search spaces.

This module contains the machine learning algorithms and their corresponding
hyperparameter grids used throughout the prostate cancer prediction pipeline.
The ModelFactory class provides a centralized interface for accessing models
and their associated hyperparameter search spaces.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import RANDOM_STATE


class ModelFactory:
    """
    Factory class for creating machine learning models and
    retrieving their hyperparameter search spaces.
    """

    def __init__(self):
        """Initialize all supported machine learning models."""

        self.models = {

            "Logistic Regression": LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ),

            "Random Forest": RandomForestClassifier(
                random_state=RANDOM_STATE
            ),

            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ),

            "SVM": SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

            "Decision Tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

            "Extra Trees": ExtraTreesClassifier(
                random_state=RANDOM_STATE
            )
        }

        self.param_grids = {

            "Logistic Regression": {
                "classifier__C": [0.01, 0.1, 1, 10]
            },

            "Random Forest": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, None]
            },

            "XGBoost": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.01, 0.1]
            },

            "SVM": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"]
            },

            "Decision Tree": {
                "classifier__max_depth": [3, 5, None]
            },

            "Extra Trees": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, None]
            }
        }

    def get_model(self, model_name):
        """
        Return a machine learning model by name.

        Parameters
        ----------
        model_name : str
            Name of the machine learning model.

        Returns
        -------
        sklearn estimator
            Initialized machine learning estimator.
        """
        return self.models[model_name]

    def get_param_grid(self, model_name):
        """
        Return the hyperparameter grid for a model.

        Parameters
        ----------
        model_name : str
            Name of the machine learning model.

        Returns
        -------
        dict
            Hyperparameter search space.
        """
        return self.param_grids[model_name]

    def get_all_models(self):
        """
        Return all available machine learning models.

        Returns
        -------
        dict
            Dictionary containing all model names and estimators.
        """
        return self.models

    def get_all_param_grids(self):
        """
        Return all hyperparameter grids.

        Returns
        -------
        dict
            Dictionary containing all parameter grids.
        """
        return self.param_grids
