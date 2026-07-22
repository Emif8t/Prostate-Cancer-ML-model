"""
main.py

Entry point for the prostate cancer machine learning pipeline.
"""

from src.dataset import Dataset
from src.preprocess import Preprocessor
from src.models import ModelFactory
from src.nested_cv import NestedCrossValidator

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    CLINICAL_FEATURES,
    MOLECULAR_FEATURES,
    COMBINED_FEATURES
)


def main():

    # =====================================================
    # Load Dataset
    # =====================================================

    print("\nLoading dataset...")

    dataset = Dataset(DATA_PATH)

    df = dataset.load()

    # =====================================================
    # Preprocess Dataset
    # =====================================================

    print("Encoding target variable...")

    df = Preprocessor.encode_group(df)

    # Update Dataset object
    dataset.df = df

    # =====================================================
    # Split Features and Target
    # =====================================================

    X_all, y = dataset.split_target(TARGET_COLUMN)

    # =====================================================
    # Create Model Factory
    # =====================================================

    model_factory = ModelFactory()

    # =====================================================
    # Create Validator
    # =====================================================

    validator = NestedCrossValidator(model_factory)

    # =====================================================
    # Feature Sets
    # =====================================================

    feature_sets = {

        "Clinical": CLINICAL_FEATURES,

        "Molecular": MOLECULAR_FEATURES,

        "Combined": COMBINED_FEATURES

    }

    all_results = {}

    all_models = {}

    # =====================================================
    # Run Models
    # =====================================================

    for feature_set_name, features in feature_sets.items():

        print("\n" + "=" * 60)

        print(f"{feature_set_name.upper()} FEATURES")

        print("=" * 60)

        X = X_all[features]

        results, models = validator.run(

            X=X,

            y=y

        )

        all_results[feature_set_name] = results

        all_models[feature_set_name] = models

    # =====================================================
    # Convenience Variables
    # =====================================================

    clinical_results = all_results["Clinical"]

    molecular_results = all_results["Molecular"]

    combined_results = all_results["Combined"]

    clinical_models = all_models["Clinical"]

    molecular_models = all_models["Molecular"]

    combined_models = all_models["Combined"]

    # =====================================================
    # Save Sample Predictions
    # =====================================================

    Exporter.sample_predictions(

    combined_results,

    "Extra Trees",

    "ExtraTrees_Predictions.xlsx"

)

    Exporter.metrics_table(

    combined_results,

    "Combined_Model_Metrics.xlsx"

)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":

    main()
