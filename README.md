# Neuro-Security

EEG-based biometric authentication using the **MOABB `Lee2019_MI` dataset** and its resting-state baseline recordings.

The pipeline uses `Lee2019_MI(train_run=True, test_run=False, resting_state=True)` so it can access the resting recordings from the two Lee2019 motor-imagery sessions. The goal is subject-specific authentication using baseline EEG, not motor-imagery classification.

## Dataset

This project uses **Lee2019_MI** through MOABB.

Core dataset facts:
- **54 subjects**
- **2 sessions** on different days
- **62 EEG channels**
- **1000 Hz** sampling rate
- resting-state recordings available when `resting_state=True`

## Protocol design used in this repo

The extractor uses the MOABB resting-state runs exposed by the Lee2019 loader:
- `0preTrainRest`
- `2postTrainRest`
- `3preTestRest`
- `5postTestRest`

### Session 1 feature splits
- **train** = `0preTrainRest` + `2postTrainRest`
- **val** = `3preTestRest`
- **test** = `5postTestRest`

### Session 2 feature split
- **test** = all four resting-state runs concatenated

This gives two authentication settings:
- **Protocol 1: within-session**
  - train on session 1 train
  - choose threshold on session 1 val
  - test on session 1 test
- **Protocol 2: cross-session**
  - train on session 1 train
  - choose threshold on session 1 val
  - test on session 2 test

## Feature extraction

Each resting-state window is converted into PSD bandpower features over five bands:
- delta: 1-4 Hz
- theta: 4-8 Hz
- alpha: 8-13 Hz
- beta: 13-30 Hz
- low gamma: 30-40 Hz

With **62 EEG channels** and **5 bands**, each window becomes a **310-dimensional** feature vector.

## Default preprocessing

- pick EEG channels only
- bandpass filter: **1-40 Hz**
- average reference
- optional artifact rejection by amplitude threshold
- optional resampling flag (disabled by default)
- default window size: **2 seconds**
- default step: **2 seconds**

## File naming

Extracted features are saved like:
- `S001_S1_train.csv`
- `S001_S1_val.csv`
- `S001_S1_test.csv`
- `S001_S2_test.csv`

By default, the Lee2019 pipeline writes to dataset-specific folders so it does not overwrite older EEGMMIDB artifacts:
- `data/raw_lee2019_mi`
- `data/features_lee2019_mi`
- `models_lee2019_mi`
- `thresholds_lee2019_mi`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Everything

Run the full project pipeline with one command:

```powershell
python scripts/run_pipeline.py
```

That command automatically runs:

- `python scripts/download_data.py`
- `python scripts/extract_features.py`
- `python scripts/train_models.py`
- `python scripts/evaluate_models.py` 

## Notes

- The current script defaults still use subjects `1-40` unless you pass `--subjects` explicitly.
- Older folders such as `data/features`, `models`, and `thresholds` may still contain EEGMMIDB-era outputs and are separate from the Lee2019 defaults above.

## Recommended scope

- Do not commit raw data, extracted features, trained models, threshold JSONs, CSV summaries, or plots. They are generated artifacts and should stay local.
- With about **10 GB** free, a practical project scope is **3 to 5 subjects** at a time if you want to keep the raw Lee2019 files locally.
- A larger local cohort will likely exceed available space because each subject has two large session `.mat` files.
- If you want to test more than 5 subjects on the same machine, the safer approach is to process a small cohort at a time and delete the raw cache between runs.

## Key limitations

- The project now uses only the **resting-state portions** of `Lee2019_MI`, not the motor-imagery task labels.
- Results on a very small cohort can look artificially strong and should be presented as a feasibility study, not a definitive biometric benchmark.
- Storage is the main practical constraint for this repo right now, not model training time.