# Loan Default Predictor

## Introduction

Loan default prediction is a critical task in the financial and banking sector, enabling institutions to assess the creditworthiness of applicants before extending credit. This project presents a machine learning–based web application that predicts the likelihood of a loan applicant defaulting on their loan, based on key financial and demographic attributes.

The application is built using **Streamlit**, a Python framework for creating interactive data applications, and is powered by a classification model trained on historical loan data. The underlying machine learning pipeline evaluates multiple algorithms during training and automatically selects the best-performing model based on the F1 score, a metric that balances precision and recall — particularly important in cases involving imbalanced classes, such as loan default prediction.

This repository converts an exploratory data science notebook into a fully functional, deployable web application, making the model accessible to non-technical users through a simple form-based interface.

**Live Application:** [https://sakshi-analytics-app-sakshi-analytics-app-loan-default-predict.streamlit.app/](https://sakshi-analytics-app-sakshi-analytics-app-loan-default-predict.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How the Application Works](#how-the-application-works)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation and Local Setup](#installation-and-local-setup)
6. [Retraining the Model](#retraining-the-model)
7. [Deployment Guide](#deployment-guide)
8. [Model Evaluation and Limitations](#model-evaluation-and-limitations)
9. [Future Improvements](#future-improvements)
10. [Technology Stack](#technology-stack)
11. [License](#license)

---

## Project Overview

The objective of this project is to provide an end-to-end solution for loan default prediction, encompassing data preprocessing, model training, model selection, and deployment. The application takes applicant information as input — such as income, loan amount, credit history, and other relevant financial indicators — and outputs a prediction indicating whether the applicant is likely to default on the loan.

The project is composed of two primary components:

1. **Training Pipeline (`train.py`):** Handles data cleaning, feature encoding, feature scaling, model training, and model evaluation. Four different classification algorithms are trained and compared, and the best-performing model is persisted to disk for later use.
2. **Web Application (`app.py`):** A Streamlit-based interface that loads the trained model and preprocessing objects, collects user input through a form, and returns a real-time prediction.

---

## How the Application Works

1. The user enters applicant details into a form displayed by the Streamlit application.
2. The input data is transformed using the same encoding and scaling procedures applied during training, ensuring consistency between training and inference.
3. The preprocessed input is passed to the trained model, which outputs a prediction indicating the probability or classification of default.
4. The result is displayed to the user in a clear, interpretable format.

This design ensures that the model behaves identically in production as it did during training, as all preprocessing logic is preserved and reused via serialized objects.

---

## Project Structure

| File | Description |
|---|---|
| `train.py` | Contains the full machine learning pipeline: data cleaning, categorical encoding, feature scaling, and training of four candidate classification models. Automatically selects and saves the best-performing model based on F1 score, along with the corresponding scaler and encoders. |
| `app.py` | The Streamlit application script. Loads the serialized model and preprocessing artifacts, renders the input form, and generates predictions based on user-submitted data. |
| `loan_default_data.csv` | The historical dataset used to train and evaluate the models. |
| `requirements.txt` | A list of all Python dependencies required to run both `train.py` and `app.py`. |
| `model.pkl` | The serialized (pickled) trained machine learning model. |
| `scaler.pkl` | The serialized feature scaler used to normalize numerical input features. |
| `encoders.pkl` | The serialized encoders used to transform categorical input features into numerical representations. |
| `feature_columns.pkl` | A serialized list preserving the exact order of feature columns expected by the model during inference. |
| `model_name.pkl` | A serialized reference indicating which of the four candidate models was selected as the best performer. |

All `.pkl` files are pre-generated and included in the repository. Retraining is only required if the dataset or the preprocessing/training pipeline is modified.

---

## Prerequisites

Before setting up this project, ensure the following are installed and available:

- **Python 3.9 or later** — the runtime environment for both the training script and the web application.
- **pip** — the Python package manager, used to install dependencies.
- **Git** — required for version control and pushing the project to GitHub.
- **A GitHub account** — required for hosting the repository, which Streamlit Community Cloud uses as its deployment source.
- **A Streamlit Community Cloud account** — required for free, public deployment of the application.

---

## Installation and Local Setup

Follow these steps to run the application on your local machine before deploying it publicly.

### Step 1: Install Dependencies

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

Launch the Streamlit application using the following command:

```bash
streamlit run app.py
```

### Step 3: Verify Functionality

Streamlit will automatically open the application in your default web browser at `http://localhost:8501`. Confirm that the form loads correctly and that predictions are generated as expected before proceeding to deployment.

---

## Retraining the Model

Retraining is necessary only when the source dataset changes or when modifications are made to the preprocessing or model training logic in `train.py`. To retrain the model, run:

```bash
python train.py
```

This command will:

- Reload and clean the dataset (`loan_default_data.csv`).
- Re-apply encoding and scaling procedures.
- Retrain all four candidate models.
- Evaluate each model using the F1 score.
- Save the best-performing model and its associated preprocessing artifacts, overwriting the existing `.pkl` files.

**Important:** Always retrain and verify the model locally before redeploying the application, to ensure that the deployed version reflects the latest changes.

---

## Deployment Guide

Streamlit Community Cloud deploys applications directly from a GitHub repository. Follow the steps below to publish and deploy the project.

### Step 1: Initialize a Git Repository

Navigate to the project directory and initialize a Git repository:

```bash
cd loan_default_app
git init
git add .
git commit -m "Loan default predictor app"
```

### Step 2: Create a Remote Repository on GitHub

1. Go to [github.com/new](https://github.com/new) and create a new, empty repository.
2. Do not initialize the repository with a README, license, or `.gitignore`, as this may cause a conflict with the local repository.

### Step 3: Push the Local Repository to GitHub

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Replace `<your-username>` and `<repo-name>` with your actual GitHub username and repository name.

### Step 4: Deploy on Streamlit Community Cloud

1. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
2. Click **New app**.
3. Select the repository containing this project.
4. Set the branch to `main`.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

Streamlit will automatically install the dependencies specified in `requirements.txt` and launch the application. Upon successful deployment, a public URL will be generated in the format `https://your-app-name.streamlit.app`, which can be shared with others without requiring any local installation.

### Continuous Deployment

Streamlit Community Cloud automatically redeploys the application whenever new changes are pushed to the connected branch on GitHub. No manual redeployment steps are required.

---

## Model Evaluation and Limitations

During the training process, four candidate classification models were evaluated, and the model with the highest F1 score was selected for deployment. However, several limitations should be noted:

- **Class Imbalance:** The target variable, `Default`, is imbalanced within the dataset, meaning that instances of default are significantly less frequent than instances of non-default. This imbalance contributes to relatively modest F1 scores across all four candidate models, as classifiers trained on imbalanced data tend to favor the majority class.
- **Reliability for Production Use:** Given the current performance metrics, this application should be considered a proof-of-concept or educational tool rather than a production-grade credit decision system. Deploying this model for real-world financial decision-making without further refinement is not recommended.

---

## Future Improvements

The following enhancements are recommended to improve the robustness and reliability of the application:

- **Addressing Class Imbalance:** Techniques such as `class_weight="balanced"`, the Synthetic Minority Oversampling Technique (SMOTE), or probability threshold tuning could be applied to improve model performance on the minority class.
- **Feature Consistency Maintenance:** If the model is retrained with an updated feature set, the corresponding input fields in `app.py` must be manually updated to remain consistent with the model's expected inputs, in order to prevent runtime errors or inaccurate predictions.
- **Model Monitoring:** For applications intended to receive sustained real-world traffic, implementing monitoring for prediction drift and periodic revalidation of model performance is advisable.
- **Explainability:** Incorporating model interpretability tools, such as SHAP (SHapley Additive exPlanations), could help users understand the key factors driving individual predictions.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language for both the training pipeline and the application. |
| Streamlit | Framework used to build and serve the interactive web application. |
| scikit-learn | Machine learning library used for model training, evaluation, and preprocessing. |
| pandas | Library used for data manipulation and analysis. |

---

## License

This project is provided for educational and demonstrative purposes. Users are encouraged to review and adapt the licensing terms according to their intended use case.
