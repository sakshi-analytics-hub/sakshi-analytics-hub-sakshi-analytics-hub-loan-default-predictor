# 💸 Loan Default Predictor

> *Will they pay it back? Let the model take a guess.*

A lightweight, interactive web app that turns a loan-default prediction notebook into something anyone can click around in — fill out an applicant's details, hit submit, and get an instant risk prediction.

**🔗 Live app:** [sakshi-analytics-app-sakshi-analytics-app-loan-default-predict.streamlit.app](https://sakshi-analytics-app-sakshi-analytics-app-loan-default-predict.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-live-brightgreen)

---

## 🎯 What It Does

Feed the app an applicant's financial and demographic details, and it runs them through the best-performing model (picked from 4 candidates during training) to predict the likelihood of loan default — no notebook, no code, just a form and an answer.

---

## 📁 What's Inside

| File | What it does |
|---|---|
| 🧠 `train.py` | Reproduces the full pipeline — cleaning, encoding, scaling, and training all 4 models — then saves the best performer (by F1 score) along with the scaler/encoders it needs. |
| 🖥️ `app.py` | The Streamlit front end. Loads the saved model and serves live predictions through a simple form. |
| 📊 `loan_default_data.csv` | The dataset that powers `train.py`. |
| 📦 `requirements.txt` | Every package the app needs to run. |
| 🎁 `model.pkl` | The trained, ready-to-use model. |
| ⚙️ `scaler.pkl` | Fitted scaler for numeric features. |
| 🔤 `encoders.pkl` | Fitted encoders for categorical features. |
| 🧩 `feature_columns.pkl` | The exact feature order the model expects. |
| 🏷️ `model_name.pkl` | Which of the 4 models won. |

All the `.pkl` files are pre-generated — you only need to retrain if the data or pipeline changes.

---

## 🚀 Getting It Running

### 1. Take it for a spin locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser should pop open at `http://localhost:8501`. Give it a test drive before shipping it anywhere.

Changed the data or the pipeline? Regenerate everything with:

```bash
python train.py
```

This rebuilds every `.pkl` file — do this *before* you deploy if you've touched the data.

### 2. Ship it to GitHub

Streamlit Community Cloud deploys straight from a repo, so let's get one set up.

```bash
cd loan_default_app
git init
git add .
git commit -m "Loan default predictor app"
```

Spin up a fresh, empty repo at [github.com/new](https://github.com/new), then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 3. Go live on Streamlit Community Cloud (it's free)

1. Head to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Point it at your repo, pick the `main` branch, and set the main file to `app.py`.
4. Hit **Deploy** and grab a coffee ☕ — Streamlit installs everything and launches the app for you.

You'll land a public URL like `https://your-app-name.streamlit.app` — shareable with anyone, no install required on their end. Every future push to `main` auto-redeploys. Zero extra steps.

---

## 🔭 Where To Take This Next

- ⚖️ **Class imbalance:** the `Default` class is skewed, which drags F1 scores down across all 4 models. Before trusting this widely, look into `class_weight="balanced"`, SMOTE, or threshold tuning.
- 🧬 **New features:** if you retrain with a different feature set, remember to update the `input_df` fields in `app.py` to match — the app and model need to stay in sync.
- 📈 **Model monitoring:** worth tracking prediction drift over time if this ever sees real traffic.

---

## 🛠️ Built With

Python · Streamlit · scikit-learn · pandas

---

<p align="center"><i>Made for exploring how a notebook model becomes something people can actually use.</i></p>
