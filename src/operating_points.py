"""Detection rate at fixed false-positive budgets.   Usage: python -m src.operating_points

A SOC picks an acceptable false-alarm rate, not a 0.5 probability cut-off. This shows
the recall each model achieves at 1/2/5/10% FPR on KDDTest+ (an analysis of the
ROC curve, not a threshold tuned for deployment)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import MODELS_DIR, REPORTS_DIR
from .data import FEATURES, clean, load_train_test


def main():
    _, test = load_train_test()
    test = clean(test)
    X, y = test[FEATURES], test["is_attack"]
    rows = {}
    for name, f in {"Random Forest": "rf_binary", "Gradient Boosting": "gb_binary"}.items():
        s = joblib.load(MODELS_DIR / f"{f}.joblib").predict_proba(X)[:, 1]
        fpr, tpr, thr = roc_curve(y, s)
        rows[name] = {f"recall @ {int(b*100)}% FPR": round(float(tpr[np.searchsorted(fpr, b, 'right') - 1]), 3)
                      for b in (0.01, 0.02, 0.05, 0.10)}
    df = pd.DataFrame(rows).T
    df.to_csv(REPORTS_DIR / "operating_points.csv")
    print(df.to_markdown() if hasattr(df, "to_markdown") else df)


if __name__ == "__main__":
    main()
