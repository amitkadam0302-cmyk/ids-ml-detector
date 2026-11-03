# 🛡️ ML Network Intrusion Detector

A machine-learning network intrusion detection system (NIDS) built with **scikit-learn** on the public
**NSL-KDD** dataset. It classifies network connections as *normal* or *attack*, identifies the attack
family (DoS, Probe, R2L, U2R), explains its decisions, and ships with a **Streamlit** demo.

> **Headline finding:** models score **~99.9% F1 in cross-validation** but only **~75-79% F1 on the official
> held-out test set**, because the test set contains attack types never seen in training. This project
> measures and analyses that generalisation gap instead of hiding it.

## Results (train: `KDDTrain+`, test: official `KDDTest+`, 22,544 records)

| Model | Precision | Recall | F1 | False-positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.917 | 0.626 | 0.744 | 7.5% | 0.791 | 0.866 |
| Random Forest | 0.968 | 0.609 | 0.747 | 2.7% | 0.961 | 0.964 |
| **Gradient Boosting** | 0.969 | 0.671 | **0.793** | 2.8% | 0.961 | 0.967 |
| Isolation Forest (unsupervised, normal-only training) | 0.955 | 0.655 | 0.777 | 4.1% | 0.941 | 0.953 |

Default threshold (0.5). 5-fold CV on training data only (Random Forest): F1 = 0.9988 ± 0.0002.

**Detection rate at a fixed false-alarm budget** (what a SOC actually tunes; `python -m src.operating_points`):

| Model | 1% FPR | 2% FPR | 5% FPR | 10% FPR |
|---|---|---|---|---|
| Random Forest | 45.7% | 51.3% | 77.2% | 89.0% |
| Gradient Boosting | 47.5% | 50.8% | 81.5% | 86.0% |

**Attack-category classifier (Random Forest, 5 classes):** DoS recall 0.76, Probe 0.59, R2L **0.01**, U2R **0.01**.

Figures are in `reports/figures/` (ROC/PR curves, confusion matrices, permutation importance, SHAP).

## Key insights

1. **The CV-vs-test gap is the real story.** NSL-KDD's test set includes 17 attack types absent from training.
   Random split CV (99.9%) wildly overestimates real-world performance. Always test on a distribution-shifted set.
2. **Good ranking, poor default threshold.** ROC-AUC is 0.96 but recall at p=0.5 is only ~0.61; at a 5% FPR budget
   the same model detects ~77-81% of attacks. Thresholds should be chosen against an alert budget.
3. **Accuracy is misleading.** Overall accuracy hides near-zero recall on R2L/U2R (rare, stealthy attacks).
4. **Missed attacks are the novel ones.** Zero-detection attacks in the test set include `guess_passwd`, `mailbomb`,
   `snmpguess`, `snmpgetattack` (see `reports/per_attack_detection_rf.csv`). Several are application-layer or
   unseen in training; the flow-level features carry little signal for them.
5. **Unsupervised detection is competitive.** Isolation Forest, trained on normal traffic only, nearly matches
   the supervised models, and is the better approach when new attack types appear.
6. **Explainability agrees with security intuition.** `src_bytes`, `dst_bytes`, connection `count`, SYN-error
   rates (`serror_rate`) and the TCP `flag` (e.g. `SF` = normal completion) drive predictions: the signatures of
   floods and scans.

## Method

| Step | Detail |
|---|---|
| Data | NSL-KDD (41 features, 125,973 train / 22,544 test). Attacks grouped into DoS / Probe / R2L / U2R |
| Cleaning | Deduplicate, replace ±inf, drop constant column `num_outbound_cmds` |
| Preprocessing | `ColumnTransformer`: median impute + scale numerics, one-hot categoricals (`handle_unknown="ignore"` since test has unseen services). Fit inside a `Pipeline` → **no leakage** |
| Imbalance | `class_weight="balanced"` |
| Models | Logistic Regression (baseline), Random Forest, HistGradientBoosting, Isolation Forest |
| Metrics | Precision, recall, F1, **FPR**, ROC-AUC, PR-AUC, confusion matrices, per-attack detection |
| Explainability | Permutation importance (original features) + SHAP |

## Quick start

```bash
git clone <your-repo-url> && cd ids-ml-detector
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                  # downloads data, trains, evaluates, saves models + figures (~3-4 min)
python -m src.operating_points       # recall at fixed FPR
pytest -q                            # unit tests
streamlit run app/streamlit_app.py   # demo (click "Use bundled sample" or upload a CSV)
```

> A standalone single-file version of the whole pipeline is included as `ids_detector_single_file.py` (`python ids_detector_single_file.py train`). The full written report is `reports/Project_Report.pdf`.

## Project structure

```
ids-ml-detector/
├── src/
│   ├── config.py            paths, seed
│   ├── data.py              download, load, attack→category mapping, cleaning
│   ├── preprocessing.py     leakage-safe sklearn preprocessing
│   ├── train.py             all models, CV, multiclass, error analysis, saving
│   ├── evaluate.py          metrics (incl. FPR) and plots
│   ├── explain.py           permutation importance + SHAP
│   └── operating_points.py  recall at fixed false-positive budgets
├── app/streamlit_app.py     interactive demo
├── tests/test_pipeline.py   unit tests
├── models/                  saved .joblib models
├── reports/                 metrics (csv/json) + figures
└── data/                    raw data (auto-downloaded) + sample_flows.csv
```

## Limitations and future work

- NSL-KDD (derived from 1999 DARPA data) is dated; real traffic today is largely encrypted. Next step: repeat on
  **CIC-IDS2017 / UNSW-NB15** and test cross-dataset transfer.
- R2L/U2R detection is near zero: needs payload/host-level features or per-class cost-sensitive training.
- Threshold chosen on a fixed alert budget should be calibrated on validation data from the target network.
- Future: live capture with Scapy/Zeek → flow features, MITRE ATT&CK mapping of predicted categories, concept-drift monitoring.

## Dataset citation

M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, *"A Detailed Analysis of the KDD CUP 99 Data Set"*, IEEE CISDA, 2009.
