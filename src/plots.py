import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc

# ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set")
plt.legend()
plt.show()


# 7. SHAP Analysis (TreeExplainer with numeric conversion)

print("\nGenerating SHAP summary plot for Random Forest...")

# Train pipeline
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])
rf_pipe.fit(X_train, y_train)

# Preprocess and ensure numeric dtype
X_train_preprocessed = rf_pipe.named_steps['preprocessor'].transform(X_train)
X_train_preprocessed = X_train_preprocessed.astype(float)  # <-- Key fix
# Get feature names after preprocessing
feature_names = numeric_features + list(
    rf_pipe.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

# Create SHAP explainer for classifier only
explainer = shap.TreeExplainer(rf_pipe.named_steps['classifier'])

# Compute SHAP values
shap_values = explainer.shap_values(X_train_preprocessed)

# Plot SHAP summary for positive class
shap.summary_plot(shap_values[1], X_train_preprocessed, feature_names=feature_names)






Add plots.py
