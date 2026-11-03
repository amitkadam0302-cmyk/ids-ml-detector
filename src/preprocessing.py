"""Scikit-learn preprocessing. Everything is fit inside a Pipeline so scalers and
encoders only ever see training data (no leakage)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC_COLS

# num_outbound_cmds is constant (always 0) in NSL-KDD -> carries no information
DROP = ["num_outbound_cmds"]


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_COLS if c not in DROP]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
