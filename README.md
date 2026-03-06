# Neuro-Security


1. Download EEG data for subjects 1-40 and runs 1-2.
2. Split each recording into train, validation, and test segments.
3. Extract 320-dimensional bandpower features from each EEG window.
4. Train one Linear SVM model per subject.
5. Evaluate authentication performance and save clean JSON and CSV outputs.

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

- `download_data.py`
- `extract_features.py`
- `train_models.py`
- `evaluate_models.py`

using the defaults already set in each script.

## Optional: Run Step By Step

If you want to run the stages manually, the short versions are:

```powershell
python scripts/download_data.py
python scripts/extract_features.py
python scripts/train_models.py
python scripts/evaluate_models.py
```

## Default Behavior

With no extra flags, the pipeline uses:

- subjects `1-40`
- runs `1,2`
- train/validation/test split of `20s / 10s / 30s`
- feature windows of **2 seconds** with 2‑second step (non‑overlapping)
- spectrograms computed as PSD bandpower over the canonical bands
- resulting feature vectors are 320‑dimensional (32 channels × 5 bands)
- run 1 for training and threshold selection
- run 1 test as protocol 1
- run 2 test as protocol 2
- artifact rejection disabled by default

## How To Read The Results

Each subject JSON contains three blocks:

- `decision`: the threshold chosen from validation data
- `protocol1`: same-run test performance at that threshold
- `protocol2`: cross-run test performance at that threshold

The main cohort-level file is `thresholds/report.json`, which reports:

- mean validation EER and AUC
- mean protocol 1 FAR, FRR, and AUC
- mean protocol 2 FAR, FRR, and AUC

## Example Final Output

After a successful run, you should see files like:

```text
thresholds/report.json
thresholds/summary.csv
thresholds/user_001.json
```