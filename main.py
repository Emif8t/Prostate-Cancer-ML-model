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
