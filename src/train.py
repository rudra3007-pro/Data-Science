import os
print("CWD =", os.getcwd())
print("FILE =", __file__)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "rainfall.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("BASE_DIR =", BASE_DIR)
print("DATA_PATH =", DATA_PATH)

os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH, encoding="latin1")


X = data.drop("RainTomorrow", axis=1)
y = data["RainTomorrow"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, os.path.join(MODEL_DIR, "rain_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
