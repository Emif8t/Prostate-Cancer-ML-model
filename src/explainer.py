"""
explainer.py

Model explainability utilities.

Provides feature importance, permutation importance,
and SHAP-based explanations for trained machine learning models.
"""

import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = ()


class Explainer:
    """
    Utilities for interpreting trained machine learning models.
    """

    # ---------------------------------------------------------
    # Internal helper
    # ---------------------------------------------------------
  
    @staticmethod
    def _unwrap_model(model):
        """
        Return the underlying estimator.

        If a scikit-learn Pipeline is supplied, the final step
        (assumed to be the estimator) is returned.
        """

        if isinstance(model, Pipeline):

            return model.steps[-1][1]

        return model
      
    # ---------------------------------------------------------
    # Internal helper
    # ---------------------------------------------------------

    @staticmethod
    def _transform_features(model, X):
        """
        Apply preprocessing if a Pipeline is supplied.

        Returns
        -------
        array-like
            Transformed feature matrix.
        """

        if not isinstance(model, Pipeline):

            return X

        # No preprocessing steps
        if len(model.steps) == 1:

            return X

        # Apply every step except the final estimator
        transformer = model[:-1]

        return transformer.transform(X)

    # ---------------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------------

    @staticmethod
    def feature_importance(model, feature_names):
        """
        Return feature importances for tree-based models.
        """

        estimator = Explainer._unwrap_model(model)

        if not hasattr(estimator, "feature_importances_"):

            raise ValueError(

                f"{type(estimator).__name__} does not provide "
                "feature_importances_."

            )

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": estimator.feature_importances_

        })

        return importance.sort_values(

            "Importance",

            ascending=False

        )

    # ---------------------------------------------------------
    # Permutation Importance
    # ---------------------------------------------------------

    @staticmethod
    def permutation_importance(
        model,
        X,
        y,
        random_state=42
    ):
        """
        Compute permutation feature importance.
        """

        perm = permutation_importance(

            estimator=model,

            X=X,

            y=y,

            scoring="roc_auc",

            n_repeats=30,

            random_state=random_state,

            n_jobs=-1

        )

        importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance": perm.importances_mean,

            "Std": perm.importances_std

        })

        return importance.sort_values(

            "Importance",

            ascending=False

        )

    # ---------------------------------------------------------
    # SHAP Values
    # ---------------------------------------------------------

        # ---------------------------------------------------------
    # SHAP Values
    # ---------------------------------------------------------

    @staticmethod
    def shap_values(model, X):
        """
        Compute SHAP values using the most appropriate explainer.

        Returns
        -------
        explainer
            SHAP explainer object.

        shap_values
            SHAP Explanation object (or equivalent).

        transformed_X
            Feature matrix after pipeline preprocessing.
        """

        estimator = Explainer._unwrap_model(model)

        transformed_X = Explainer._transform_features(model, X)

        # =====================================================
        # Tree-based models
        # =====================================================

        tree_models = (
            RandomForestClassifier,
            ExtraTreesClassifier,
            DecisionTreeClassifier,
        )

        if XGBClassifier:
            tree_models = tree_models + (XGBClassifier,)

        if isinstance(estimator, tree_models):

            explainer = shap.TreeExplainer(estimator)

            shap_values = explainer(transformed_X)

            return explainer, shap_values, transformed_X

        # =====================================================
        # Linear models
        # =====================================================

        if isinstance(estimator, LogisticRegression):

            explainer = shap.LinearExplainer(
                estimator,
                transformed_X
            )

            values = explainer.shap_values(transformed_X)

            if isinstance(values, list):
                values = values[1]

            shap_values = shap.Explanation(
                values=values,
                base_values=np.repeat(
                    explainer.expected_value,
                    transformed_X.shape[0]
                ),
                data=transformed_X,
                feature_names=X.columns.tolist()
            )

            return explainer, shap_values, transformed_X

        # =====================================================
        # General fallback
        # =====================================================

        background = shap.sample(
            transformed_X,
            min(100, len(transformed_X)),
            random_state=42
        )

        if hasattr(estimator, "predict_proba"):

            predict_fn = lambda data: estimator.predict_proba(data)[:, 1]

        else:

            predict_fn = estimator.predict

        explainer = shap.KernelExplainer(
            predict_fn,
            background
        )

        values = explainer.shap_values(transformed_X)

        if isinstance(values, list):
            values = values[1]

        shap_values = shap.Explanation(
            values=values,
            base_values=np.repeat(
                explainer.expected_value,
                transformed_X.shape[0]
            ),
            data=transformed_X,
            feature_names=X.columns.tolist()
        )

        return explainer, shap_values, transformed_X

    # ---------------------------------------------------------
    # SHAP Summary
    # ---------------------------------------------------------

    @staticmethod
    def shap_summary(model, X):

        _, shap_values, transformed_X = Explainer.shap_values(

            model,

            X

        )

        shap.summary_plot(

            shap_values,

            transformed_X,

            feature_names=X.columns,

            show=True

        )

    # ---------------------------------------------------------
    # SHAP Bar
    # ---------------------------------------------------------

    @staticmethod
    def shap_bar(model, X):

        _, shap_values, _ = Explainer.shap_values(

            model,

            X

        )

        shap.plots.bar(

            shap_values

        )

    # ---------------------------------------------------------
    # SHAP Waterfall
    # ---------------------------------------------------------

    @staticmethod
    def shap_waterfall(
        model,
        X,
        sample_index=0
    ):

        _, shap_values, _ = Explainer.shap_values(

            model,

            X

        )

        shap.plots.waterfall(

            shap_values[sample_index]

        )

    # ---------------------------------------------------------
    # SHAP Dependence
    # ---------------------------------------------------------

    @staticmethod
    def shap_dependence(
        model,
        X,
        feature
    ):

        _, shap_values, transformed_X = Explainer.shap_values(

            model,

            X

        )

        shap.dependence_plot(

            feature,

            shap_values.values,

            transformed_X,

            feature_names=X.columns

        )
