import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.metrics import roc_curve, auc

for name, model_pipe in calibrated_pipelines.items():
    print(f"\nPlotting curves for {name}")
 

#ROC Curve (all models)
plt.figure(figsize=(7,7))
for name, model in calibrated_pipelines.items():
    y_prob = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_val = roc_auc_score(y, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#Calibration Curve (all models)
plt.figure(figsize=(7,7))
for name, model in calibrated_pipelines.items():
    y_prob = model.predict_proba(X)[:,1]
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    # LOWESS smooth
    smooth = lowess(prob_true, prob_pred, frac=0.6)
    plt.plot(smooth[:,0], smooth[:,1], marker='o', label=name)

plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Probability")
plt.title("Calibration Curve - All Models")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#Decision Curve Analysis (all models)
thresholds = np.linspace(0.01,0.99,99)
N = len(y)
plt.figure(figsize=(7,5))

for name, model in calibrated_pipeliness.items():
    y_prob = model.predict_proba(X)[:,1]
    net_benefit = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        nb = (tp/N) - (fp/N) * (t/(1-t))
        net_benefit.append(nb)
    plt.plot(thresholds, net_benefit, label=name)

# Treat-all and Treat-none
prevalence = y.mean()
nb_treatall = prevalence - (1 - prevalence)*(thresholds/(1-thresholds))
plt.plot(thresholds, nb_treatall, '--', label='Treat all')
plt.plot(thresholds, np.zeros_like(thresholds), ':', label='Treat none')

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis - All Models")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#SHAP Plot
!pip install shap

def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Extract transformed feature names from ColumnTransformer."""
    output_features = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            output_features.extend(columns)
        elif name == "cat":
            ohe = transformer
            cats = ohe.categories_
            new_cols = []
            for col, cat_list in zip(categorical_cols, cats):
                new_cols.extend([f"{col}_{c}" for c in cat_list])
            output_features.extend(new_cols)
    return output_features

feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
print("Total features:", len(feature_names))


cal_model = calibrated_pipelines["XGBoost"]  # choose your best model
clf = cal_model.named_steps["classifier"]

base_clf = clf.calibrated_classifiers_[0].estimator


preprocessor = cal_model.named_steps["preprocessor"]
X_transformed = preprocessor.transform(X)

if not isinstance(X_transformed, np.ndarray):
    X_transformed = X_transformed.toarray()

X_transformed = X_transformed.astype(np.float32)

feature_names = preprocessor.get_feature_names_out()


if "Forest" in str(type(base_clf)) or "Boost" in str(type(base_clf)) or "XGB" in str(type(base_clf)):
    explainer = shap.TreeExplainer(base_clf)
else:
    explainer = shap.LinearExplainer(base_clf, X_transformed)


shap_values = explainer.shap_values(X_transformed)


shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)


Add plots.py
