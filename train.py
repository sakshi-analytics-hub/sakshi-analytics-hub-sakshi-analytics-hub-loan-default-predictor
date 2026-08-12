"""
Trains the same 4 models as the notebook, picks the best one by F1 score,
and saves everything the app needs to make predictions: the trained model,
the scaler, and the label encoders.

Run this once (or whenever you retrain):  python train.py
"""

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "loan_default_data.csv"

# ---- load & clean (same steps as the notebook) ----
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=["Purpose_Description"])

data["Annual_Income"] = data["Annual_Income"].fillna(data["Annual_Income"].median())
data["Credit_Score"] = data["Credit_Score"].fillna(data["Credit_Score"].median())

le_employment = LabelEncoder()
le_education = LabelEncoder()
data["Employment_Type"] = le_employment.fit_transform(data["Employment_Type"])
data["Education"] = le_education.fit_transform(data["Education"])

X = data.drop(columns=["Default"])
y = data["Default"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append(
        {
            "name": name,
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
    )
    print(f"{name}: acc={results[-1]['accuracy']:.3f} f1={results[-1]['f1']:.3f}")

best = max(results, key=lambda r: r["f1"])
print(f"\nBest model by F1 score: {best['name']} (f1={best['f1']:.3f})")

# ---- save everything the app needs ----
joblib.dump(best["model"], "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(
    {"Employment_Type": le_employment, "Education": le_education},
    "encoders.pkl",
)
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(best["name"], "model_name.pkl")

print("\nSaved: model.pkl, scaler.pkl, encoders.pkl, feature_columns.pkl, model_name.pkl")
