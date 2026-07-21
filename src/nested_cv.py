import numpy as np

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)


ci = BootstrapCI.auc(y, y_prob_all)

class NestedCrossValidator:

    def __init__(self):

        self.outer_cv = ...

        self.inner_cv = ...


_evaluate_model()

for train_idx, test_idx in outer_cv.split(X, y):

            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            pipeline = build_pipeline(model)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_prob = best_model.predict_proba(X_test)[:,1]

            y_prob_all[test_idx] = y_prob

            sample_predictions.loc[test_idx, "Predicted_Probability"] = y_prob

_train_final_model()
    final_grid.fit(X, y)


_calculate_metrics()

 y_pred = (y_prob_all >= 0.5).astype(int)

        sample_predictions["Predicted_Class"] = y_pred

        sample_predictions["Correct"] = (
            sample_predictions["True_Label"] ==
            sample_predictions["Predicted_Class"]
        )

        ci = bootstrap_auc_ci(y, y_prob_all)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
       
        results[name] = {

            # Predictions
            "y_prob": y_prob_all,
            "y_pred": y_pred,\

            "Sample_Predictions": sample_predictions,

            # NEW: save feature names
            "feature_names": X.columns.tolist(),

            "Best_Params": best_params,
            
            # Performance
            "AUC": ci[1],
            "AUC_Low": ci[0],
            "AUC_High": ci[2],

            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1": f1_score(y, y_pred),
            "Brier": brier_score_loss(y, y_prob_all),

            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp)
        }

    return results, final_models
