# Neuro-Security

EEG-based biometric authentication using the **MOABB `Lee2019_MI` dataset** and its resting-state baseline recordings.

For the pipeline, we uses `Lee2019_MI(train_run=True, test_run=False, resting_state=True)` so it can access the resting recordings from the two Lee2019 motor-imagery sessions. The goal is subject-specific authentication using baseline EEG, not motor-imagery classification.

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

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1\
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