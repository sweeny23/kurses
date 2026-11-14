# Ancient Structures Classification Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24223317.svg)](https://doi.org/10.5281/zenodo.24223317)

This repository contains two machine learning pipelines built to analyze and classify historical structures using the `ancient_structures_dataset.csv`. The scripts evaluate multiple ensemble classifiers, perform repeated stratified cross-validation, and generate detailed performance reports including feature importance and confusion matrices.

## Repository Contents

The project is split into two distinct classification tasks:

* **`ancient_ml_2.py` (Purpose Prediction):** Predicts the **category** of an ancient structure (e.g., defensive, funerary, infrastructure) based purely on geographic coordinates and location precision.
* **`ml.py` (Material Prediction):** Predicts the construction **material** or method. It uses the geographic coordinates plus the structure's known `category` as an additional feature, operating on the premise that knowing *what* a building is helps deduce *how* it was built.

Both scripts include an optional `--use-text` flag that incorporates a TF-IDF vectorizer to analyze the structure's name and description, significantly boosting accuracy.

## Methodology

Both scripts run a comprehensive evaluation pipeline:
1. **Data Preprocessing:** Loads the dataset, merges minority classes into an "other" category, and encodes the target labels.
2. **Model Comparison:** Evaluates a suite of ensemble models (Random Forest, Extra Trees, and optionally XGBoost, LightGBM, CatBoost, and a custom Voting Classifier) using 5-fold repeated stratified cross-validation.
3. **Evaluation Metric:** Models are primarily ranked by their **Macro F1 Score** to account for class imbalances.
4. **Testing & Reporting:** The top-performing model is trained on an 80% split and tested on a 20% held-out set.
5. **Feature Importance:** Permutation importance is calculated to determine which features the model relied on most heavily.

## Requirements

Ensure you have the following core libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```
*Optional but recommended for the full model suite:*
```bash
pip install xgboost lightgbm catboost
```

## Usage

### Running via Command Line (Terminal)

You can run either script directly from your terminal:

**Run the Category Predictor (Geo features only):**
```bash
python ancient_ml_2.py --data ancient_structures_dataset.csv --out results_category
```

**Run the Material Predictor (Geo + Category features, plus Text):**
```bash
python ml.py --data ancient_structures_dataset.csv --out results_material --use-text
```

### Running in a Jupyter/Kaggle Notebook

```python
import ancient_ml_2
import ml

# 1. Predict structure categories
category_results = ancient_ml_2.run(
    data_path='ancient_structures_dataset.csv', 
    out_dir='./category_results'
)

# 2. Predict construction materials (with text features enabled)
material_results = ml.run(
    data_path='ancient_structures_dataset.csv', 
    out_dir='./material_results', 
    use_text=True
)
```

## Outputs

For every run, the script generates a folder containing:
* `model_comparison.csv`: Cross-validation scores for all evaluated models.
* `classification_report.csv`: Precision, recall, and F1-scores for the test split.
* `confusion_matrix.png`: Visual matrix of correct and confused predictions.
* `feature_importance.csv` & `feature_importance.png`: Permutation importance of each feature.
* `best_model.joblib`: The serialized final model pipeline ready for deployment.

* 
