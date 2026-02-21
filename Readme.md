# Credit Risk Prediction

A machine-learning web app that predicts whether a loan applicant is a **good** or **bad** credit risk, based on the German Credit dataset.

## Project Structure

```
credit-risk-project/
├── app/
│   └── app.py          # Streamlit front-end
├── src/
│   └── train.py        # Model training script
├── data/
│   └── raw/
│       └── german_credit_data.csv
├── models/             # Saved model & encoders (.pkl)
├── notebooks/
│   └── analysis_model.ipynb
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app/app.py
```

## Retraining the Model

```bash
python src/train.py
```
