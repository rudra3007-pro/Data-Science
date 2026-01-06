from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "rain_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["pressure"]),
        float(request.form["wind"]),
        float(request.form["cloud"])
    ]

    sample = scaler.transform([values])
    prediction = model.predict(sample)[0]

    result = "🌧 Rain Tomorrow" if prediction == 1 else "☀ No Rain Tomorrow"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
