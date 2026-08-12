# Loan Default Predictor — Streamlit App

A small web app version of your loan default notebook. Enter an applicant's
details and get an instant prediction.

## Files
- `train.py` — reproduces your notebook's pipeline (cleaning, encoding,
  scaling, training all 4 models) and saves the best one (by F1 score) plus
  the scaler/encoders it needs.
- `app.py` — the Streamlit app that loads those saved files and serves
  predictions through a form.
- `loan_default_data.csv` — your dataset (needed to run `train.py`).
- `requirements.txt` — Python packages needed to run the app.
- `model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl`,
  `model_name.pkl` — already generated for you by `train.py`. You don't need
  to retrain unless you change the data or pipeline.

## 1. Test it locally first
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`. Confirm it
works before deploying.

If you ever change the data or the pipeline, retrain with:
```bash
python train.py
```
This regenerates the `.pkl` files — remember to re-run this before deploying
if you make changes.

## 2. Put the project on GitHub
Streamlit Community Cloud deploys straight from a GitHub repo.

```bash
cd loan_default_app
git init
git add .
git commit -m "Loan default predictor app"
```
Create a new empty repo on GitHub (github.com/new), then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud (free)
1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Pick your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.

That's it — Streamlit installs your `requirements.txt` and launches the app.
You'll get a public URL like `https://your-app-name.streamlit.app` that
anyone can open, no installation needed on their end.

## Notes / things worth improving later
- Your dataset's `Default` class is imbalanced, so F1 scores across all 4
  models are fairly low (the notebook picks the best of a weak bunch). If you
  want a genuinely reliable predictor before sharing this widely, consider
  handling class imbalance (e.g. `class_weight="balanced"`, SMOTE, or
  threshold tuning) before deploying to real users.
- If you retrain later with different features, update the `input_df` fields
  in `app.py` to match.
- Any time you push a new commit to GitHub, Streamlit Community Cloud
  auto-redeploys — no extra steps needed.
