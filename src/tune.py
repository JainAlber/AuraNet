"""
AuraNet — Optuna hyperparameter tuning for XGBoost.

Optimises macro F1-score (better than accuracy for class-imbalanced security data)
over 60 trials using a TPE sampler with a MedianPruner for early trial stopping.

Parameters searched
-------------------
  n_estimators      100 – 600
  max_depth           3 – 10
  learning_rate    0.01 – 0.30  (log scale)
  subsample        0.50 – 1.00
  colsample_bytree 0.50 – 1.00
  min_child_weight    1 – 10
  gamma            0.00 – 0.50

Usage
-----
  python src/tune.py            # uses data/processed_train.csv
  python src/tune.py <path>     # custom processed CSV
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise

PROCESSED_PATH = Path("data/processed_train.csv")
EXPORTS        = Path("exports")
MODELS_DIR     = Path("models")
EXPORTS.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

N_TRIALS    = 60
CV_FOLDS    = 3
RANDOM_SEED = 42


# ── data ─────────────────────────────────────────────────────────────────────

def load(path: Path):
    df = pd.read_csv(path)
    y  = (df["label"] == "Attack").astype(int)
    X  = df.drop(columns=["label"])
    return X, y


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_tr, y_tr):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"     : trial.suggest_int   ("n_estimators",      100, 600),
            "max_depth"        : trial.suggest_int   ("max_depth",           3,  10),
            "learning_rate"    : trial.suggest_float ("learning_rate",    0.01, 0.30, log=True),
            "subsample"        : trial.suggest_float ("subsample",         0.5,  1.0),
            "colsample_bytree" : trial.suggest_float ("colsample_bytree",  0.5,  1.0),
            "min_child_weight" : trial.suggest_int   ("min_child_weight",    1,  10),
            "gamma"            : trial.suggest_float ("gamma",             0.0,  0.5),
            "eval_metric"      : "logloss",
            "random_state"     : RANDOM_SEED,
            "n_jobs"           : -1,
        }

        fold_f1s = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            model = XGBClassifier(**params)
            model.fit(
                X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
                eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])],
                verbose=False,
            )
            y_pred = model.predict(X_tr.iloc[val_idx])
            fold_f1s.append(f1_score(y_tr.iloc[val_idx], y_pred, average="macro"))

            # Optuna pruning: report intermediate score after each fold
            trial.report(np.mean(fold_f1s), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_f1s))

    return objective


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_optimisation_history(study: optuna.Study) -> None:
    trials = [t for t in study.trials if t.value is not None]
    values = [t.value for t in trials]
    best   = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(len(values)), values, s=18, alpha=0.55, color="#4C9BE8", label="Trial F1")
    ax.plot(range(len(best)),   best,    lw=2,  color="#E8574C", label="Best so far")
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Macro F1  (3-fold CV)")
    ax.set_title("Optuna Optimisation History — XGBoost F1")
    ax.legend()
    fig.tight_layout()
    out = EXPORTS / "optuna_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Optimisation history  -> {out}")


def plot_param_importance(study: optuna.Study) -> None:
    importances = optuna.importance.get_param_importances(study)
    names  = list(importances.keys())
    scores = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names[::-1], scores[::-1], color="#4C9BE8", edgecolor="white")
    for bar, val in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (fANOVA)")
    ax.set_title("Optuna Hyperparameter Importances")
    fig.tight_layout()
    out = EXPORTS / "optuna_param_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Param importance      -> {out}")


def plot_confusion_matrix(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    total = cm.sum()
    for text_obj, (val, pct) in zip(
        ax.texts, [(v, v / total * 100) for row in cm for v in row]
    ):
        text_obj.set_text(f"{val}\n({pct:.1f}%)")
        text_obj.set_fontsize(12)
    ax.set_title("Confusion Matrix — Tuned XGBoost (NSL-KDD)")
    fig.tight_layout()
    out = EXPORTS / "confusion_matrix_tuned.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix      -> {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROCESSED_PATH

    print(f"[LOAD] {path}")
    X, y = load(path)
    print(f"       {len(X):,} rows  |  features: {X.shape[1]}")
    print(f"       Normal: {(y==0).sum():,}  |  Attack: {(y==1).sum():,}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"[TUNE] Starting Optuna study  ({N_TRIALS} trials, {CV_FOLDS}-fold CV) ...")
    print(       "       Optimising: macro F1-score\n")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Progress callback — prints a dot per completed trial
    completed = [0]
    def _progress(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed[0] += 1
            marker = f"  Trial {completed[0]:>3}/{N_TRIALS}  F1={trial.value:.5f}  best={study.best_value:.5f}"
            print(marker)

    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS, callbacks=[_progress])

    # ── Best params ───────────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print("  BEST HYPERPARAMETERS  (Optuna TPE / macro F1)")
    print(f"{'='*54}")
    for k, v in study.best_params.items():
        print(f"  {k:<22} {v}")
    print(f"  {'CV macro F1':<22} {study.best_value:.5f}")
    print(f"{'='*54}\n")

    # ── Final model — retrain on full train set with best params ──────────────
    print("[MODEL] Retraining final model on full train set ...")
    best_params = {**study.best_params, "eval_metric": "logloss",
                   "random_state": RANDOM_SEED, "n_jobs": -1}
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    # ── Evaluation on held-out test set ───────────────────────────────────────
    y_pred = final_model.predict(X_test)
    print(f"\n{'='*54}")
    print("  FINAL TEST-SET RESULTS  (Tuned XGBoost — NSL-KDD)")
    print(f"{'='*54}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=4))

    # ── Persist model + training report ──────────────────────────────────────
    model_path = MODELS_DIR / "xgb_tuned.joblib"
    joblib.dump(final_model, model_path)
    print(f"[SAVE] Model saved -> {model_path}")

    report = {
        "best_f1_cv":    study.best_value,
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_trials":      N_TRIALS,
        "n_completed":   sum(1 for t in study.trials if t.value is not None),
        "n_train":       len(X_train),
        "n_features":    int(X.shape[1]),
        "best_params":   study.best_params,
    }
    report_path = MODELS_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[SAVE] Training report -> {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_optimisation_history(study)
    plot_param_importance(study)
    plot_confusion_matrix(final_model, X_test, y_test)

    print("\n[DONE] All artefacts saved to exports/ and models/\n")


if __name__ == "__main__":
    main()
