import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# Cross-validation results

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Cross-validated ROC-AUC scores:")
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    print(f"{name} - ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Train & Evaluate

plt.figure(figsize=(8, 6))
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
# Predictions on test set
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"\n{name} - Test Set Metrics")
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Precision:", precision_score(y_test, y_pred))
    print("Test Recall:", recall_score(y_test, y_pred))
    print("Test F1 Score:", f1_score(y_test, y_pred))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))




Add model.py
