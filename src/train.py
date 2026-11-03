"""Train + evaluate all models.   Usage:  python -m src.train

Protocol: train on KDDTrain+, evaluate on the official KDDTest+ (which contains
attack types absent from training), so scores reflect generalisation, not memorisation.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from .data import CLASS_ORDER, FEATURES, clean, load_train_test
from .explain import permutation_plot, shap_summary
from .preprocessing import build_preprocessor


def make_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced",
                                           random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE))]),
    }


def attack_score(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train, test = load_train_test()
    n_raw = len(train)
    train, test = clean(train), clean(test)
    print(f"train {n_raw} -> {len(train)} after cleaning | test {len(test)}")
    ev.plot_class_distribution(train, test)

    Xtr, ytr = train[FEATURES], train["is_attack"]
    Xte, yte = test[FEATURES], test["is_attack"]
    results, scores = {}, {}

    # ---- 1. Supervised binary models -------------------------------------
    models = make_models()
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        pred, s = pipe.predict(Xte), attack_score(pipe, Xte)
        results[name] = ev.binary_metrics(yte, pred, s)
        scores[name] = s
        print(f"{name:22s} F1={results[name]['f1']:.3f} FPR={results[name]['false_positive_rate']:.3f}")

    # ---- 2. Isolation Forest: unsupervised, trained on NORMAL traffic only ----
    prep = build_preprocessor(scale=True).fit(Xtr[ytr == 0])
    iso = IsolationForest(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(prep.transform(Xtr[ytr == 0]))
    normal_scores = -iso.score_samples(prep.transform(Xtr[ytr == 0]))   # higher = more anomalous
    thr = np.quantile(normal_scores, 0.95)                              # target ~5% FPR on normal
    s_iso = -iso.score_samples(prep.transform(Xte))
    results["Isolation Forest (unsupervised)"] = ev.binary_metrics(yte, (s_iso > thr).astype(int), s_iso)
    scores["Isolation Forest (unsupervised)"] = s_iso
    joblib.dump({"prep": prep, "model": iso, "threshold": float(thr)}, MODELS_DIR / "isolation_forest.joblib")
    print(f"{'Isolation Forest':22s} F1={results['Isolation Forest (unsupervised)']['f1']:.3f}")

    ev.plot_curves(scores, yte)
    res_df = pd.DataFrame(results).T
    res_df.to_csv(REPORTS_DIR / "binary_results.csv")
    json.dump(results, open(REPORTS_DIR / "binary_results.json", "w"), indent=2)

    # ---- 3. Cross-validation on training data (stability check) ------------
    cv = cross_validate(make_models()["Random Forest"], Xtr, ytr,
                        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                        scoring=["f1", "precision", "recall"], n_jobs=1)
    cv_summary = {k.replace("test_", ""): [round(float(cv[k].mean()), 4), round(float(cv[k].std()), 4)]
                  for k in cv if k.startswith("test_")}
    json.dump(cv_summary, open(REPORTS_DIR / "cv_random_forest.json", "w"), indent=2)
    print("5-fold CV (train only, RF):", cv_summary)

    # ---- 4. Multiclass attack-category classifier --------------------------
    mc = make_models()["Random Forest"]
    mc.fit(Xtr, train["category"])
    mc_pred = mc.predict(Xte)
    report = classification_report(test["category"], mc_pred, labels=CLASS_ORDER,
                                   output_dict=True, zero_division=0)
    json.dump(report, open(REPORTS_DIR / "multiclass_report.json", "w"), indent=2)
    pd.DataFrame(report).T.round(3).to_csv(REPORTS_DIR / "multiclass_report.csv")
    ev.plot_confusion(test["category"], mc_pred, CLASS_ORDER,
                      "Attack category (row-normalised recall)", "confusion_multiclass.png")
    print(classification_report(test["category"], mc_pred, labels=CLASS_ORDER, zero_division=0))

    # ---- 5. Confusion matrix + explainability for the best binary model ------
    best = max((n for n in models), key=lambda n: results[n]["f1"])
    ev.plot_confusion(yte, models[best].predict(Xte), [0, 1], f"{best}: normal(0) vs attack(1)",
                      "confusion_binary.png")
    imp = permutation_plot(models["Random Forest"], Xte, yte)
    imp.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    shap_ok = shap_summary(models["Random Forest"], Xte)
    print("Top features:", list(imp.feature.head(5)), "| SHAP:", shap_ok)

    # ---- 6. Error analysis: which specific attacks get missed? ---------------
    rf_pred = models["Random Forest"].predict(Xte)
    err = (test.assign(pred=rf_pred)[test.is_attack == 1]
           .groupby("label").agg(n=("pred", "size"), detected=("pred", "mean"))
           .query("n >= 20").sort_values("detected"))
    err["detected"] = err["detected"].round(3)
    err.to_csv(REPORTS_DIR / "per_attack_detection_rf.csv")

    # ---- 7. Persist deployable models -----------------------------------------
    joblib.dump(models["Random Forest"], MODELS_DIR / "rf_binary.joblib", compress=3)
    joblib.dump(mc, MODELS_DIR / "rf_multiclass.joblib", compress=3)
    joblib.dump(models["Gradient Boosting"], MODELS_DIR / "gb_binary.joblib", compress=3)
    test.sample(300, random_state=RANDOM_STATE)[FEATURES].to_csv(
        REPORTS_DIR.parent / "data" / "sample_flows.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s. Best binary model: {best}")


if __name__ == "__main__":
    main()
