"""Loading and labelling the NSL-KDD dataset."""
from __future__ import annotations

import urllib.request
import numpy as np
import pandas as pd

from .config import BASE_URL, RAW_DIR, TEST_FILE, TRAIN_FILE

CATEGORICAL = ["protocol_type", "service", "flag"]

NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Original column order of the NSL-KDD files
FEATURES = (
    ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
    + NUMERIC[3:]
)
COLUMNS = FEATURES + ["label", "difficulty"]

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


def download(force: bool = False) -> None:
    """Download NSL-KDD train/test files if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_FILE, TEST_FILE):
        if force or not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(BASE_URL + path.name, path)


def _load(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["category"] = df["label"].map(ATTACK_CATEGORY)
    if df["category"].isna().any():
        unknown = df.loc[df["category"].isna(), "label"].unique()
        raise ValueError(f"Unmapped attack labels: {unknown}")
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    download()
    return _load(TRAIN_FILE), _load(TEST_FILE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and replace inf with NaN (none expected in NSL-KDD,
    but this keeps the pipeline safe on messier datasets such as CIC-IDS2017)."""
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates(subset=FEATURES + ["label"])
    return df.reset_index(drop=True)


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make an uploaded CSV look like model input (used by the Streamlit app).

    Accepts files with a header, or headerless NSL-KDD style files with
    41 (features), 42 (+label) or 43 (+label+difficulty) columns.
    """
    if not set(FEATURES).issubset(df.columns):
        if df.shape[1] in (41, 42, 43):
            df = df.copy()
            df.columns = COLUMNS[: df.shape[1]]
        else:
            missing = sorted(set(FEATURES) - set(df.columns))
            raise ValueError(f"CSV is missing required columns, e.g. {missing[:5]}")
    X = df[FEATURES].copy()
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


NUMERIC_COLS = [c for c in FEATURES if c not in CATEGORICAL]
