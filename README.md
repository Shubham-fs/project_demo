# Knowledge Gate Conversion Intelligence

Lead scoring project for predicting which students are most likely to purchase a paid course based on their product behavior on the website.

This project is designed for sales prioritization, not just raw classification. The model produces calibrated purchase probabilities, converts them into business-friendly lead tiers, and exposes the results through a Streamlit dashboard.

## What This Project Does

- Scores each student with a calibrated purchase probability
- Separates leads into:
  - `HIGH`: high-confidence likely buyers
  - `MID`: students still worth outreach based on economics
  - `LOW`: nurture rather than immediate sales focus
- Generates a daily sales sheet for unconverted students
- Provides holdout evaluation, lift charts, decile analysis, and feature explanations

## Current Model Snapshot

Latest completed local training run:

- Holdout ROC-AUC: `0.9658`
- Holdout Average Precision: `0.9125`
- Top 10% lift: `4.00x`
- Top 20% buyer capture: `70.8%`
- High-confidence threshold: `0.421`
- Outreach threshold: `0.023`

Interpretation:

- `HIGH` leads are tuned for confidence and achieved about `80.5%` precision on the holdout set
- `MID` leads are above the outreach threshold and are suitable for broader outreach

## Project Structure

- `train.py`: trains the calibrated XGBoost model and saves artifacts
- `app.py`: Streamlit dashboard for evaluation and lead scoring
- `kg_features.py`: shared feature engineering used in both training and scoring
- `data_validation.py`: schema and quality validation for training data and uploaded files
- `models/`: saved model, preprocessor, calibrator, and config

## How The Modeling Pipeline Works

1. Raw training data is validated before modeling starts.
2. Shared feature engineering adds behavior-derived signals.
3. Known leaky post-purchase fields are removed.
4. Multiple XGBoost candidates are compared with stratified cross-validation.
5. The selected model is trained on the training split.
6. Probabilities are calibrated with isotonic regression on a separate calibration split.
7. Final thresholds are learned from calibration data:
   - `HIGH` threshold targets high precision
   - `MID` threshold is used for broader outreach coverage
8. The exact holdout student IDs are saved so evaluation remains reproducible.

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train The Model

```powershell
python train.py
```

This writes:

- `models/kg_model.pkl`
- `models/kg_preprocessor.pkl`
- `models/kg_calibrator.pkl`
- `models/kg_config.pkl`

## Run The Dashboard

```powershell
streamlit run app.py
```

## Deploy

The easiest way to deploy this project is as a Streamlit app.

Deployment-ready files already included in this repo:

- `app.py` as the app entrypoint
- `requirements.txt` in the repo root
- `.streamlit/config.toml` for headless app settings
- pre-trained artifacts inside `models/`

### Recommended Path: Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from that repository.
4. Set the entrypoint file to `app.py`.
5. Deploy.

If the platform asks for dependencies, keep using the root `requirements.txt`.

Notes:

- The app already includes the trained model files, so deployment does not require retraining first.
- SHAP was kept optional in the code and removed from deployment dependencies to make cloud deployment lighter. The app will still work without SHAP explanations.
- The dataset file and model artifacts are loaded from local relative paths, so they should remain in the repository when deploying.

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Data Validation

Validation now checks:

- required columns
- duplicate or blank `student_id`
- binary columns must contain only `0/1`
- key numeric fields must be non-negative
- percentage/range fields must stay within sensible limits

Uploaded scoring files are checked before predictions are generated.

## Important Notes

- This is a strong project-grade system and a good interview/demo artifact.
- It is not a full production deployment yet.
- For real deployment, add scheduled retraining, drift monitoring, CI, secret handling, and service/API packaging.
