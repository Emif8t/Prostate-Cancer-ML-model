from src.dataset import Dataset

from src.models import ModelFactory

from src.nested_cv import NestedCrossValidator

from src.visualization import Visualizer

from src.exporter import Exporter

def main():
    print("📥 Loading data...")
    X, y = load_data()

    print("🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print("🤖 Training model...")
    model = train_model(X_train, y_train)

    print("📊 Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"✅ Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()


factory = ModelFactory()

validator = NestedCrossValidator(factory)

results, models = validator.run(X, y)


model_factory = ModelFactory()
validator = NestedCrossValidator(model_factory)

feature_sets = {
    "Clinical": clinical_features,
    "Molecular": molecular_features,
    "Combined": combined_features
}

all_results = {}
all_models = {}

for name, features in feature_sets.items():

    print(f"\n==== {name.upper()} ====")

    results, models = validator.run(
        X=df[features],
        y=y
    )

    all_results[name] = results
    all_models[name] = models

clinical_results = all_results["Clinical"]
molecular_results = all_results["Molecular"]
combined_results = all_results["Combined"]

clinical_models = all_models["Clinical"]
molecular_models = all_models["Molecular"]
combined_models = all_models["Combined"]

combined_results["Extra Trees"]["Sample_Predictions"].to_excel(
    "ExtraTrees_Predictions.xlsx",
    index=False
)
