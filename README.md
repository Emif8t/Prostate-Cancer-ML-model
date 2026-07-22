# Prostate Cancer Machine Learning Model

A modular Python package for developing and evaluating supervised machine learning models for prostate cancer prediction using clinical and molecular biomarkers.

The package was developed as part of a PhD research project investigating molecular markers for prostate cancer in African men. It implements a reproducible machine learning workflow including data preprocessing, nested cross-validation, hyperparameter tuning, model evaluation, explainable artificial intelligence (XAI), visualization, and result export.

---

## Features

* Modular and object-oriented architecture
* Clinical, molecular, and combined feature modelling
* Nested stratified cross-validation
* Hyperparameter optimization using GridSearchCV
* Bootstrap confidence intervals for ROC AUC
* Comprehensive performance metrics
* ROC and calibration curve visualization
* Explainability using SHAP
* Permutation and tree-based feature importance
* Export of predictions and evaluation metrics
* Easily extensible for additional datasets and models

---

## Machine Learning Models

The package currently supports:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Support Vector Machine (SVM)
* Extreme Gradient Boosting (XGBoost)

---

## Project Structure

```text
Prostate-Cancer-ML-model/
│
├── data/
│   └── ROCdata2.xlsx
│
├── src/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── config.py
│   ├── dataset.py
│   ├── explainer.py
│   ├── exporter.py
│   ├── metrics.py
│   ├── models.py
│   ├── nested_cv.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── utils.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Prostate-Cancer-ML-model.git
```

Move into the project directory:

```bash
cd Prostate-Cancer-ML-model
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Required Python Packages

* numpy
* pandas
* scikit-learn
* matplotlib
* shap
* xgboost
* openpyxl

Optional:

* jupyter

---

## Input Dataset

The input dataset should contain one row per participant and include the target variable together with the selected predictor variables.

Example columns:

| Variable         | Description                       |
| ---------------- | --------------------------------- |
| PSA              | Prostate-specific antigen         |
| age              | Participant age                   |
| Expression_ASS1  | ASS1 gene expression              |
| Expression_CPS1  | CPS1 gene expression              |
| Methylation_ASS1 | ASS1 methylation                  |
| Methylation_CPS1 | CPS1 methylation                  |
| SNP_1            | Genetic variant                   |
| SNP_2            | Genetic variant                   |
| SNP_3            | Genetic variant                   |
| Group            | Target variable (Case or Control) |

The feature sets used by the package are defined in `src/config.py`.

---

## Running the Pipeline

Execute the main script:

```bash
python main.py
```

The workflow performs the following steps:

1. Load the dataset.
2. Encode the target variable.
3. Split predictors and target.
4. Build machine learning pipelines.
5. Perform nested cross-validation.
6. Optimize model hyperparameters.
7. Evaluate model performance.
8. Calculate bootstrap confidence intervals.
9. Generate visualizations.
10. Export predictions and performance metrics.

---

## Package Workflow

```text
Dataset
    │
    ▼
Preprocessor
    │
    ▼
Pipeline Builder
    │
    ▼
Nested Cross Validation
    │
    ▼
Hyperparameter Optimization
    │
    ▼
Model Evaluation
    │
    ├── Metrics
    ├── Bootstrap Confidence Intervals
    ├── Visualization
    ├── Explainability (SHAP)
    └── Export Results
```

---

## Performance Metrics

The package computes:

* Area Under the ROC Curve (AUC)
* Accuracy
* Precision
* Recall
* F1-score
* Sensitivity
* Specificity
* Brier Score

---

## Explainable Artificial Intelligence

The package includes model interpretation tools through SHAP and permutation importance.

Available explainability functions include:

* SHAP Summary Plot
* SHAP Bar Plot
* SHAP Waterfall Plot
* SHAP Dependence Plot
* Tree-Based Feature Importance
* Permutation Feature Importance

---

## Outputs

Depending on the selected workflow, the package can generate:

* ROC curves
* Calibration curves
* SHAP visualizations
* Feature importance tables
* Bootstrap AUC estimates
* Sample-level predictions
* Performance summary tables
* Best hyperparameters

---

## Configuration

Most configurable settings are stored in `src/config.py`, including:

* Dataset path
* Target variable
* Clinical feature list
* Molecular feature list
* Combined feature list
* Number of outer folds
* Number of inner folds
* Bootstrap iterations
* Random seed
* Probability threshold

---

## Extending the Package

The modular design makes it straightforward to:

* Add new machine learning algorithms.
* Incorporate additional biomarkers.
* Apply the workflow to other diseases.
* Integrate external validation datasets.
* Include new explainability methods.

---

## Citation

If you use this software in your research, please cite the associated publication once available.

---

## Author

**Emmanuel Israel**

PhD Candidate, Biochemistry

Research interests:

* Biochemistry
* Artificial Intelligence in Healthcare
* Machine Learning
* Precision Medicine
* Cancer Genomics
* Prostate Cancer Biomarkers
* Bioinformatics

---

## License

This project is released under the MIT License.
