# Titanic ML Deployment and Inference Strategies

## Dataset
Download `train.csv` from the official Kaggle Titanic competition:
https://www.kaggle.com/competitions/titanic/data

Place `train.csv` in this project directory. This assignment uses only that dataset.

## Setup (Windows / VS Code)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run in order
```powershell
python train_model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
```
Training saves `model.pkl`; batch inference saves `batch_predictions.csv`.

## Start the API
In terminal 1:
```powershell
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs. Test `GET /health` and `POST /predict`.
Try a valid payload and then an invalid payload (e.g. `Pclass: 8`) to observe Pydantic validation.

In terminal 2:
```powershell
python client.py
```

## Model design
A single scikit-learn Pipeline includes missing-value imputation, scaling/one-hot encoding, and Random Forest classification. The pipeline is saved once and reused by every inference script and the API. The held-out test portion is used for inference demonstrations and is not used for fitting the model.

## Required screenshots
Capture:
1. Training accuracy and saved model
2. Online inference output
3. Batch output and `batch_predictions.csv`
4. Streaming output showing 10 records
5. Swagger `/health` success
6. Swagger valid `/predict` success
7. Swagger invalid `/predict` validation error

## GitHub
Create a public repository, upload all project files and generated artifacts, and submit the repository URL through your faculty's Google Form. Do not upload secrets.
