"""Explainability: permutation importance (model-agnostic) + optional SHAP."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FIG_DIR, RANDOM_STATE
from .data import FEATURES


def permutation_plot(pipe, X, y, top=15, fname="feature_importance.png") -> pd.DataFrame:
    """Importance on ORIGINAL features (one-hot columns are not split up)."""
    sample = X.sample(min(5000, len(X)), random_state=RANDOM_STATE)
    res = permutation_importance(pipe, sample, y.loc[sample.index], scoring="f1",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": FEATURES, "importance": res.importances_mean,
                         "std": res.importances_std})
           .sort_values("importance", ascending=False))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t.feature, t.importance, xerr=t["std"], color="#2b6cb0")
    ax.set_xlabel("Drop in F1 when feature is shuffled")
    ax.set_title("Permutation importance (binary Random Forest, KDDTest+)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    return imp


def shap_summary(pipe, X, n=500, fname="shap_summary.png") -> bool:
    """SHAP summary for the attack class. Returns False if shap isn't usable."""
    try:
        import numpy as np
        import shap
        pre, model = pipe.named_steps["prep"], pipe.named_steps["clf"]
        sample = X.sample(min(n, len(X)), random_state=RANDOM_STATE)
        Xt = pre.transform(sample)
        names = pre.get_feature_names_out()
        sv = shap.TreeExplainer(model).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=names, max_display=15, show=False)
        plt.title("SHAP values - class 'attack'")
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:  # pragma: no cover
        print(f"[warn] SHAP skipped: {e}")
        return False
