"""
AuraNet training & analysis pipeline.

Stages
------
1. Load processed_train.csv
2. RandomForest  -> feature importance bar chart
3. Correlation heatmap (numerical features)
4. XGBoost baseline -> Accuracy / Precision / Recall / F1
5. Confusion matrix plot

All charts saved to exports/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROCESSED_PATH = Path("data/processed_train.csv")
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

PALETTE = {"Normal": "#4C9BE8", "Attack": "#E8574C"}
FEATURE_COLS = [
    "duration", "protocol_type", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "network_intensity",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(path: Path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS]
    y = (df["label"] == "Attack").astype(int)   # 0 = Normal, 1 = Attack
    return X, y, df


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── 1. RandomForest feature importance ───────────────────────────────────────

def rf_feature_importance(X_train, y_train, X: pd.DataFrame) -> pd.Series:
    print("\n[RF] Training RandomForestClassifier for feature importance ...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\n  Feature Importances (RandomForest):")
    for feat, score in importance.items():
        bar = "#" * int(score * 60)
        print(f"  {feat:<22} {score:.4f}  {bar}")

    return importance


def plot_feature_importance(importance: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E8574C" if f == "network_intensity" else "#4C9BE8" for f in importance.index]
    bars = ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1], edgecolor="white")

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("RandomForest Feature Importances\n(red = engineered feature: network_intensity)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    for bar, val in zip(bars, importance.values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)

    fig.tight_layout()
    out = EXPORTS / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT] Feature importance saved -> {out}")


# ── 2. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = FEATURE_COLS + ["label_enc"]
    heatmap_df = df[FEATURE_COLS].copy()
    heatmap_df["label_enc"] = (df["label"] == "Attack").astype(int)

    corr = heatmap_df.corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle hidden
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor="#333",
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap  (label_enc: 0=Normal, 1=Attack)", fontsize=12)
    fig.tight_layout()
    out = EXPORTS / "correlation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Correlation heatmap saved  -> {out}")


# ── 3. XGBoost baseline ───────────────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n[XGB] Training XGBoost baseline ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return xgb


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str) -> None:
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n{'='*52}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"{'='*52}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    # annotate each cell with count + percentage
    total = cm.sum()
    for text_obj, (val, pct) in zip(
        ax.texts, [(v, v / total * 100) for row in cm for v in row]
    ):
        text_obj.set_text(f"{val}\n({pct:.1f}%)")
        text_obj.set_fontsize(12)

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    fig.tight_layout()
    out = EXPORTS / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix saved     -> {out}")


# ── 5. network_intensity ablation ────────────────────────────────────────────

def network_intensity_ablation(X_train, X_test, y_train, y_test) -> None:
    """
    Trains a second XGBoost WITHOUT network_intensity to quantify the
    feature's contribution to F1.
    """
    drop_col = "network_intensity"
    cols = [c for c in X_train.columns if c != drop_col]

    xgb_no_ni = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb_no_ni.fit(X_train[cols], y_train, verbose=False)
    y_pred_no_ni = xgb_no_ni.predict(X_test[cols])

    f1_no_ni = f1_score(y_test, y_pred_no_ni)
    print(f"\n[ABLATION] XGBoost WITHOUT network_intensity -> F1: {f1_no_ni:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[LOAD] Reading {PROCESSED_PATH} ...")
    X, y, df = load_data(PROCESSED_PATH)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # --- feature importance
    importance = rf_feature_importance(X_train, y_train, X)
    plot_feature_importance(importance)

    # --- correlation heatmap
    plot_correlation_heatmap(df)

    # --- XGBoost baseline
    xgb = train_xgboost(X_train, X_test, y_train, y_test)
    evaluate(xgb, X_test, y_test, "XGBoost Baseline")
    plot_confusion_matrix(xgb, X_test, y_test, "XGBoost Baseline")

    # --- ablation: does network_intensity help?
    network_intensity_ablation(X_train, X_test, y_train, y_test)

    print("\n[DONE] All artefacts saved to exports/\n")


if __name__ == "__main__":
    main()
