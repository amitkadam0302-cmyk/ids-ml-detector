"""Streamlit demo.   Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, ROOT
from src.data import coerce_features

st.set_page_config(page_title="ML Network Intrusion Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    return (joblib.load(MODELS_DIR / "rf_binary.joblib"),
            joblib.load(MODELS_DIR / "rf_multiclass.joblib"))


st.title("🛡️ ML Network Intrusion Detector")
st.caption("Random Forest trained on NSL-KDD. Upload connection records to flag suspicious traffic.")

try:
    binary, multi = load_models()
except FileNotFoundError:
    st.error("Models not found. Run `python -m src.train` first.")
    st.stop()

threshold = st.sidebar.slider("Alert threshold (attack probability)", 0.05, 0.95, 0.50, 0.05,
                              help="Lower = catch more attacks but more false alarms.")
up = st.sidebar.file_uploader("Upload CSV of connection records", type=["csv"])
use_sample = st.sidebar.button("Use bundled sample (300 flows)")

df = None
if up is not None:
    df = pd.read_csv(up)
elif use_sample:
    df = pd.read_csv(ROOT / "data" / "sample_flows.csv")

if df is None:
    st.info("Upload a CSV (NSL-KDD format, 41 features; header optional) or click the sample button.")
    st.stop()

try:
    X = coerce_features(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

p_attack = binary.predict_proba(X)[:, 1]
out = df.copy()
out["attack_probability"] = p_attack.round(4)
out["alert"] = p_attack >= threshold
out["predicted_category"] = multi.predict(X)
out.loc[~out["alert"], "predicted_category"] = "normal"

c1, c2, c3 = st.columns(3)
c1.metric("Records scanned", len(out))
c2.metric("Alerts raised", int(out.alert.sum()))
c3.metric("Alert rate", f"{out.alert.mean():.1%}")

left, right = st.columns(2)
left.subheader("Alerts by predicted category")
left.bar_chart(out.loc[out.alert, "predicted_category"].value_counts())
right.subheader("Attack-probability distribution")
right.bar_chart(pd.cut(out.attack_probability, 10).value_counts().sort_index().rename(str))

st.subheader("Flagged records (most suspicious first)")
cols = ["attack_probability", "predicted_category", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
cols = [c for c in cols if c in out.columns]
st.dataframe(out[out.alert].sort_values("attack_probability", ascending=False)[cols])
st.download_button("Download all predictions (CSV)", out.to_csv(index=False), "predictions.csv", "text/csv")
