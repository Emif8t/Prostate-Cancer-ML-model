"""
pipeline.py

Pipeline construction for machine learning models.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PipelineBuilder:
    """
    Builds preprocessing and machine learning pipelines.
    """

    @staticmethod
    def build(model):
        """
        Create a Scikit-learn pipeline consisting of
        feature scaling followed by a classifier.

        Parameters
        ----------
        model : sklearn estimator
            Machine learning model.

        Returns
        -------
        Pipeline
            Configured Scikit-learn pipeline.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
