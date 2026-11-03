#!/usr/bin/env python3
"""
ML Network Intrusion Detector  (single-file version)
====================================================
Classifies network connections from the public NSL-KDD dataset as normal / attack
using scikit-learn, and identifies the attack family (DoS, Probe, R2L, U2R).

Usage
-----
    pip install -r requirements.txt
    python ids_detector.py train                 # download data, train, evaluate, save models/figures
    python ids_detector.py predict flows.csv     # score a CSV of connection records
    python ids_detector.py predict               # score a 300-flow sample from the test set

Protocol: train on KDDTrain+, evaluate on the official KDDTest+ (contains attack types
absent from training), so scores reflect generalisation, not memorisation.
Outputs go to ./models and ./reports (created next to this file).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
RAW_DIR, MODELS_DIR = ROOT / "data" / "raw", ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TRAIN_FILE, TEST_FILE = RAW_DIR / "KDDTrain+.txt", RAW_DIR / "KDDTest+.txt"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
SEED = 42

# --------------------------------------------------------------------------- data
CATEGORICAL = ["protocol_type", "service", "flag"]
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]
NUMERIC_COLS = [c for c in FEATURES if c not in CATEGORICAL]
DROP = ["num_outbound_cmds"]          # constant column in NSL-KDD
CLASS_ORDER = ["normal", "DoS", "Probe", "R2L", "U2R"]

ATTACK_CATEGORY = {"normal": "normal"}
for _cat, _names in {
    "DoS": "back land neptune pod smurf teardrop mailbomb apache2 processtable udpstorm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster "
           "sendmail named snmpgetattack snmpguess xlock xsnoop worm",
    "U2R": "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm",
}.items():
    for _n in _names.split():
        ATTACK_CATEGORY[_n] = _cat


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_FILE, TEST_FILE):
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(BASE_URL + path.name, path)


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    if df["category"].isna().any():
        raise ValueError(f"Unmapped labels: {df.loc[df.category.isna(), 'label'].unique()}")
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, replace +-inf with NaN (safe for messier datasets too)."""
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates(subset=FEATURES + ["label"])
    return df.reset_index(drop=True)


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Accept CSVs with a header, or headerless NSL-KDD files with 41/42/43 columns."""
    if not set(FEATURES).issubset(df.columns):
        if df.shape[1] in (41, 42, 43):
            df = df.copy()
            df.columns = COLUMNS[: df.shape[1]]
        else:
            raise ValueError(f"CSV is missing required columns, e.g. {sorted(set(FEATURES) - set(df.columns))[:5]}")
    X = df[FEATURES].copy()
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# --------------------------------------------------------------------------- preprocessing / models
def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Fit inside a Pipeline, so scalers/encoders only see training data (no leakage)."""
    num_cols = [c for c in NUMERIC_COLS if c not in DROP]
    steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer(
        [("num", Pipeline(steps), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="drop", verbose_feature_names_out=False)


def make_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("prep", build_preprocessor(True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", build_preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", build_preprocessor(False)),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]),
    }


# --------------------------------------------------------------------------- evaluation
def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {"precision": precision_score(y_true, y_pred, zero_division=0),
           "recall (detection rate)": recall_score(y_true, y_pred),
           "f1": f1_score(y_true, y_pred),
           "false_positive_rate": fp / (fp + tn), "false_negative_rate": fn / (fn + tp),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=labels,
        normalize="true", cmap="Blues", ax=ax, values_format=".2f", colorbar=False)
    ax.set_title(title)
    _save(fig, fname)


def plot_curves(scores: dict, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        p, r, _ = precision_recall_curve(y_true, s)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
        axes[1].plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (KDDTest+)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (KDDTest+)")
    for a in axes:
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
    _save(fig, "roc_pr_curves.png")


def plot_class_distribution(train, test):
    x, w = np.arange(len(CLASS_ORDER)), 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, [(train.category == c).mean() * 100 for c in CLASS_ORDER], w, label="KDDTrain+")
    ax.bar(x + w / 2, [(test.category == c).mean() * 100 for c in CLASS_ORDER], w, label="KDDTest+")
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_ylabel("% of records")
    ax.set_title("Class distribution (note the train/test shift)")
    ax.legend()
    _save(fig, "class_distribution.png")


# --------------------------------------------------------------------------- explainability
def permutation_plot(pipe, X, y, top=15) -> pd.DataFrame:
    sample = X.sample(min(5000, len(X)), random_state=SEED)
    res = permutation_importance(pipe, sample, y.loc[sample.index], scoring="f1",
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": res.importances_mean,
                        "std": res.importances_std}).sort_values("importance", ascending=False)
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t.feature, t.importance, xerr=t["std"], color="#2b6cb0")
    ax.set_xlabel("Drop in F1 when feature is shuffled")
    ax.set_title("Permutation importance (Random Forest, KDDTest+)")
    _save(fig, "feature_importance.png")
    return imp


def shap_summary(pipe, X, n=500) -> bool:
    try:
        import shap
        pre, model = pipe.named_steps["prep"], pipe.named_steps["clf"]
        sample = X.sample(min(n, len(X)), random_state=SEED)
        Xt = pre.transform(sample)
        sv = shap.TreeExplainer(model).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=pre.get_feature_names_out(), max_display=15, show=False)
        plt.title("SHAP values - class 'attack'")
        plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"[warn] SHAP skipped: {e}")
        return False


# --------------------------------------------------------------------------- training
def train() -> None:
    t0 = time.time()
    for d in (MODELS_DIR, REPORTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    download()
    train_df, test_df = clean(load_dataset(TRAIN_FILE)), clean(load_dataset(TEST_FILE))
    print(f"train {len(train_df)} | test {len(test_df)}")
    plot_class_distribution(train_df, test_df)
    Xtr, ytr, Xte, yte = train_df[FEATURES], train_df["is_attack"], test_df[FEATURES], test_df["is_attack"]
    results, scores = {}, {}

    # 1. supervised binary models
    models = make_models()
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        s = pipe.predict_proba(Xte)[:, 1]
        results[name], scores[name] = binary_metrics(yte, pipe.predict(Xte), s), s
        print(f"{name:22s} F1={results[name]['f1']:.3f}  FPR={results[name]['false_positive_rate']:.3f}")

    # 2. Isolation Forest: unsupervised, fit on NORMAL traffic only
    prep = build_preprocessor(True).fit(Xtr[ytr == 0])
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(prep.transform(Xtr[ytr == 0]))
    thr = np.quantile(-iso.score_samples(prep.transform(Xtr[ytr == 0])), 0.95)   # ~5% FPR on normal
    s_iso = -iso.score_samples(prep.transform(Xte))
    key = "Isolation Forest (unsupervised)"
    results[key], scores[key] = binary_metrics(yte, (s_iso > thr).astype(int), s_iso), s_iso
    joblib.dump({"prep": prep, "model": iso, "threshold": float(thr)}, MODELS_DIR / "isolation_forest.joblib")
    print(f"{'Isolation Forest':22s} F1={results[key]['f1']:.3f}")

    plot_curves(scores, yte)
    pd.DataFrame(results).T.to_csv(REPORTS_DIR / "binary_results.csv")

    # 3. 5-fold CV on training data (shows how optimistic random-split CV is)
    cv = cross_validate(make_models()["Random Forest"], Xtr, ytr,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring=["f1", "precision", "recall"])
    cv_summary = {k[5:]: [round(float(v.mean()), 4), round(float(v.std()), 4)] for k, v in cv.items() if k.startswith("test_")}
    json.dump(cv_summary, open(REPORTS_DIR / "cv_random_forest.json", "w"), indent=2)
    print("5-fold CV (RF, train only):", cv_summary)

    # 4. multiclass attack-family classifier
    mc = make_models()["Random Forest"].fit(Xtr, train_df["category"])
    mc_pred = mc.predict(Xte)
    print(classification_report(test_df["category"], mc_pred, labels=CLASS_ORDER, zero_division=0))
    pd.DataFrame(classification_report(test_df["category"], mc_pred, labels=CLASS_ORDER, output_dict=True,
                 zero_division=0)).T.round(3).to_csv(REPORTS_DIR / "multiclass_report.csv")
    plot_confusion(test_df["category"], mc_pred, CLASS_ORDER, "Attack family (row-normalised recall)", "confusion_multiclass.png")

    # 5. best model confusion matrix + explainability
    best = max(models, key=lambda n: results[n]["f1"])
    plot_confusion(yte, models[best].predict(Xte), [0, 1], f"{best}: normal(0) vs attack(1)", "confusion_binary.png")
    permutation_plot(models["Random Forest"], Xte, yte).to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    shap_summary(models["Random Forest"], Xte)

    # 6. error analysis: which attacks are missed?
    err = (test_df.assign(pred=models["Random Forest"].predict(Xte))[test_df.is_attack == 1]
           .groupby("label").agg(n=("pred", "size"), detected=("pred", "mean")).query("n >= 20").sort_values("detected"))
    err.round(3).to_csv(REPORTS_DIR / "per_attack_detection_rf.csv")

    # 7. detection rate at fixed false-positive budgets
    rows = {}
    for name in ("Random Forest", "Gradient Boosting"):
        fpr, tpr, _ = roc_curve(yte, scores[name])
        rows[name] = {f"recall @ {int(b * 100)}% FPR": round(float(tpr[np.searchsorted(fpr, b, 'right') - 1]), 3)
                      for b in (0.01, 0.02, 0.05, 0.10)}
    pd.DataFrame(rows).T.to_csv(REPORTS_DIR / "operating_points.csv")
    print(pd.DataFrame(rows).T)

    # 8. persist models + sample file
    joblib.dump(models["Random Forest"], MODELS_DIR / "rf_binary.joblib", compress=3)
    joblib.dump(mc, MODELS_DIR / "rf_multiclass.joblib", compress=3)
    test_df.sample(300, random_state=SEED)[FEATURES].to_csv(ROOT / "sample_flows.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s. Best binary model: {best}")


# --------------------------------------------------------------------------- inference
def predict(csv_path: str | None = None, threshold: float = 0.5) -> pd.DataFrame:
    try:
        binary, multi = joblib.load(MODELS_DIR / "rf_binary.joblib"), joblib.load(MODELS_DIR / "rf_multiclass.joblib")
    except FileNotFoundError:
        sys.exit("Models not found. Run: python ids_detector.py train")
    raw = pd.read_csv(csv_path or ROOT / "sample_flows.csv")
    X = coerce_features(raw)
    out = raw.copy()
    out["attack_probability"] = binary.predict_proba(X)[:, 1].round(4)
    out["alert"] = out["attack_probability"] >= threshold
    out["predicted_category"] = np.where(out["alert"], multi.predict(X), "normal")
    out.to_csv(ROOT / "predictions.csv", index=False)
    print(f"{len(out)} records scanned, {int(out.alert.sum())} alerts ({out.alert.mean():.1%}).")
    print(out.loc[out.alert, "predicted_category"].value_counts().to_string())
    print("Saved predictions.csv")
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 2 and sys.argv[1] == "predict":
        predict(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
