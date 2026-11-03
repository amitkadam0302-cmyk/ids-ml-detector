import numpy as np
import pandas as pd
import pytest

from src.data import ATTACK_CATEGORY, COLUMNS, FEATURES, clean, coerce_features
from src.preprocessing import build_preprocessor


def _fake(n=20):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    df["protocol_type"], df["service"], df["flag"] = "tcp", "http", "SF"
    return df


def test_feature_count():
    assert len(FEATURES) == 41 and len(COLUMNS) == 43


def test_attack_mapping_covers_core_attacks():
    for a in ["neptune", "smurf", "ipsweep", "guess_passwd", "buffer_overflow"]:
        assert a in ATTACK_CATEGORY


def test_preprocessor_ignores_unseen_category():
    X = _fake()
    prep = build_preprocessor().fit(X)
    X2 = X.copy()
    X2["service"] = "never_seen_service"
    assert prep.transform(X2).shape[0] == len(X2)


def test_clean_removes_duplicates_and_inf():
    df = _fake(5)
    df["label"] = "normal"
    df = pd.concat([df, df.iloc[[0]]])
    df.loc[df.index[1], "src_bytes"] = np.inf
    out = clean(df)
    assert len(out) == 5 and np.isfinite(out.select_dtypes("number").fillna(0).values).all()


def test_coerce_headerless_csv():
    df = _fake(3)
    raw = pd.DataFrame(df.values)  # no header, 41 cols
    assert list(coerce_features(raw).columns) == FEATURES


def test_coerce_rejects_bad_csv():
    with pytest.raises(ValueError):
        coerce_features(pd.DataFrame({"a": [1]}))
