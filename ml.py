"""
ancient_ml.py - ensemble classifiers for the ancient structures dataset.

What it does
------------
1. Loads ancient_structures_dataset.csv.
2. Builds features and the target (`material`; classes with fewer than
   `min_class_size` rows are merged into "other").
3. Compares several ensemble classifiers with repeated stratified
   cross-validation (accuracy, balanced accuracy, macro F1).
4. Trains the best model on a stratified train split, evaluates it on a
   held-out test split, and writes: a classification report, a confusion
   matrix image, permutation feature importances, and the saved model.

Feature sets
------------
default  : latitude, longitude, location_precision, max_accuracy_m, category
           (Now includes 'category' since knowing what a building is helps predict how it was built)
--use-text : adds TF-IDF of name + description. 

Run on Kaggle (notebook cell):
    import ancient_ml
    results = ancient_ml.run()                    # geo + category features only
    results = ancient_ml.run(use_text=True)       # with name/description text
Run from a terminal:
    python ancient_ml.py --data ancient_structures_dataset.csv --out results
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

DATA_NAME = "ancient_structures_dataset.csv"
TARGET = "material"  # Changed from 'category' to predict construction material/method
NUMERIC = ["latitude", "longitude", "max_accuracy_m"]
CATEGORICAL = ["location_precision", "category"]  # 'category' is now a feature
TEXT_COLUMN = "text"  # built from name + description


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def find_data(path: str | None = None) -> Path:
    """Return the dataset path: explicit path, Kaggle input, or current dir."""
    if path:
        return Path(path)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = list(kaggle.rglob(DATA_NAME))
        if hits:
            return hits[0]
    for candidate in (Path(DATA_NAME), Path("/mnt/user-data/outputs") / DATA_NAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {DATA_NAME}. Pass data_path=... or attach the dataset."
    )


def load_data(path: str | None = None, min_class_size: int = 20):
    """Load the CSV and return (X dataframe, encoded y, LabelEncoder)."""
    df = pd.read_csv(find_data(path))
    counts = df[TARGET].value_counts()
    df["target"] = df[TARGET].where(df[TARGET].map(counts) >= min_class_size, "other")
    df[TEXT_COLUMN] = (df["name"].astype(str) + " " + df["description"].astype(str))
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["target"])
    return df, y, encoder


# --------------------------------------------------------------------------
# features and models
# --------------------------------------------------------------------------
def build_preprocessor(use_text: bool) -> ColumnTransformer:
    transformers = [
        ("num", "passthrough", NUMERIC),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CATEGORICAL,
        ),
    ]
    if use_text:
        transformers.append(
            (
                "txt",
                TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=2, max_features=400
                ),
                TEXT_COLUMN,
            )
        )
    return ColumnTransformer(transformers)


def get_models(use_text: bool, seed: int = 42) -> dict:
    """Return {name: estimator}. Optional libraries are added if installed."""
    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    models = {"RandomForest": rf, "ExtraTrees": et}

    if use_text:
        # sparse text matrix: keep models that accept sparse input
        lr = LogisticRegression(max_iter=2000, C=5.0)
        models["LogisticRegression"] = lr
        models["Voting(RF+ET+LR)"] = VotingClassifier(
            [("rf", rf), ("et", et), ("lr", lr)], voting="soft"
        )
    else:
        hgb = HistGradientBoostingClassifier(random_state=seed)
        models["HistGradientBoosting"] = hgb
        try:
            from lightgbm import LGBMClassifier

            models["LightGBM"] = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1
            )
        except ImportError:
            pass
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=300, learning_rate=0.05, random_state=seed, verbosity=0
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier

            models["CatBoost"] = CatBoostClassifier(
                iterations=300, random_seed=seed, verbose=0, thread_count=-1
            )
        except ImportError:
            pass
        models["Voting(RF+ET+HGB)"] = VotingClassifier(
            [("rf", rf), ("et", et), ("hgb", hgb)], voting="soft"
        )
    return models


def make_pipeline(estimator, use_text: bool) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(use_text)), ("clf", estimator)])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def compare_models(df, y, use_text=False, seed=42, n_splits=5, n_repeats=3):
    """Repeated stratified CV for every model; returns a sorted DataFrame."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    rows = []
    for name, est in get_models(use_text, seed).items():
        pipe = make_pipeline(est, use_text)
        res = cross_validate(pipe, df, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "accuracy": res["test_accuracy"].mean(),
                "accuracy_std": res["test_accuracy"].std(),
                "balanced_accuracy": res["test_balanced_accuracy"].mean(),
                "f1_macro": res["test_f1_macro"].mean(),
            }
        )
        print(f"  {name:22s} acc={rows[-1]['accuracy']:.3f} "
              f"(+/-{rows[-1]['accuracy_std']:.3f})  f1_macro={rows[-1]['f1_macro']:.3f}")
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    return table.reset_index(drop=True)


def train_and_report(df, y, encoder, best_name, use_text, out_dir: Path, seed=42):
    """Fit the best model on a train split, report on the held-out test split."""
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed
    )
    x_train, x_test = df.iloc[idx_train], df.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    pipe = make_pipeline(get_models(use_text, seed)[best_name], use_text)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)

    labels = encoder.classes_
    report = classification_report(
        y_test, pred, target_names=labels, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(out_dir / "classification_report.csv")
    print("\nHeld-out test report:")
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_test, pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # permutation importance on the original input columns
    imp = permutation_importance(
        pipe, x_test, y_test, n_repeats=15, random_state=seed, scoring="f1_macro"
    )
    imp_df = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
            }
        )
        .query("feature in @NUMERIC or feature in @CATEGORICAL or feature == @TEXT_COLUMN")
        .sort_values("importance", ascending=False)
    )
    if not use_text:  # text column is not a model input in geo-only mode
        imp_df = imp_df[imp_df["feature"] != TEXT_COLUMN]
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], xerr=imp_df["std"][::-1])
    ax.set_xlabel("Drop in macro F1 when shuffled")
    ax.set_title("Permutation importance (test split)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # refit on all data for the saved model
    final = make_pipeline(get_models(use_text, seed)[best_name], use_text).fit(df, y)
    joblib.dump(
        {"model": final, "label_encoder": encoder, "use_text": use_text},
        out_dir / "best_model.joblib",
    )
    return report_df, imp_df


def predict(df_new: pd.DataFrame, model_path: str) -> pd.Series:
    """Predict categories for new rows (needs the same columns as the dataset)."""
    bundle = joblib.load(model_path)
    df_new = df_new.copy()
    df_new[TEXT_COLUMN] = df_new["name"].astype(str) + " " + df_new["description"].astype(str)
    codes = bundle["model"].predict(df_new)
    return pd.Series(bundle["label_encoder"].inverse_transform(codes), index=df_new.index)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def run(data_path=None, out_dir=None, use_text=False, seed=42, n_repeats=3,
        min_class_size=20):
    """Full pipeline. Returns a dict with the CV table, reports and best model name."""
    if out_dir is None:
        out_dir = "/kaggle/working" if Path("/kaggle/working").exists() else "results"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, y, encoder = load_data(data_path, min_class_size)
    print(f"Rows: {len(df)} | classes: {list(encoder.classes_)}")
    print(f"Majority-class baseline accuracy: {np.bincount(y).max() / len(y):.3f}")
    print(f"Feature set: {'geo + category + text' if use_text else 'geo + category only'}\n")

    print("Cross-validation (5 folds x %d repeats):" % n_repeats)
    table = compare_models(df, y, use_text, seed, n_repeats=n_repeats)
    table.to_csv(out / "model_comparison.csv", index=False)
    print("\n", table.round(3).to_string(index=False))

    best = table.loc[0, "model"]
    print(f"\nBest by macro F1: {best} (differences between top models are often within noise)")
    report_df, imp_df = train_and_report(df, y, encoder, best, use_text, out, seed)
    print(f"\nSaved to: {out.resolve()}")
    return {"comparison": table, "report": report_df, "importance": imp_df, "best": best}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", default=None, help="path to the CSV")
    parser.add_argument("--out", default=None, help="output folder")
    parser.add_argument("--use-text", action="store_true", help="add name/description TF-IDF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    args, _ = parser.parse_known_args(argv)  # tolerant of notebook arguments
    run(args.data, args.out, args.use_text, args.seed, args.repeats)


if __name__ == "__main__":
    main()