import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10]
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 5]
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.01, 0.1]
    }
}

# TRAIN–TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# CALIBRATION
calibrated_pipelines = {}

for name, model in models.items():
    print(f"\nTraining model: {name}")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    best_clf = best_pipe.named_steps['classifier']

    ### Calibration
    calibrated_classifier = CalibratedClassifierCV(best_clf, cv=5, method="sigmoid")

    calibrated_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_classifier)
    ])

    calibrated_pipeline.fit(X_train, y_train)

    calibrated_pipelines[name] = calibrated_pipeline

#Bootstrpping (95% CI)

def bootstrap_metrics(model_pipe, X, y, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(X)

    metrics = {
        "AUC": [], "Accuracy": [], "Precision": [],
        "Recall": [], "F1": [], "Brier": [],
        "Sensitivity": [], "Specificity": []
    }

    desc_name = model_pipe.named_steps["classifier"].__class__.__name__

    for i in trange(B, desc=f"Bootstrapping {desc_name}"):
        idx = rng.choice(np.arange(N), size=N, replace=True)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        preds = model_pipe.predict(Xb)
        probs = model_pipe.predict_proba(Xb)[:, 1]

        auc = roc_auc_score(yb, probs)
        acc = accuracy_score(yb, preds)
        pre = precision_score(yb, preds)
        rec = recall_score(yb, preds)
        f1 = f1_score(yb, preds)
        brier = brier_score_loss(yb, probs)

        cm = confusion_matrix(yb, preds)
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp+fn) > 0 else 0
        spec = tn / (tn + fp) if (tn+fp) > 0 else 0

        metrics["AUC"].append(auc)
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(pre)
        metrics["Recall"].append(rec)
        metrics["F1"].append(f1)
        metrics["Brier"].append(brier)
        metrics["Sensitivity"].append(sens)
        metrics["Specificity"].append(spec)

    return metrics

bootstrap_results = {}

for name, model_pipe in calibrated_pipelines.items():
    print(f"\nBootstrapping: {name}")
    res = bootstrap_metrics(model_pipe, X, y, B=2000)
    bootstrap_results[name] = res

# REPORT 95% CI
def ci(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)

print("\n95% CONFIDENCE INTERVALS\n")

for name, res in bootstrap_results.items():
    print(f"\nMODEL: {name}")
    for metric, vals in res.items():
        low, high = ci(vals)
        print(f"{metric}: {np.mean(vals):.3f}  (95% CI {low:.3f} – {high:.3f})")



Add model.py
