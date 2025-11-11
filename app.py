from flask import Flask, render_template, request
import numpy as np
import pickle
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -------- Feature list --------
feature_names = [
    "Temperature (°C)",
    "Humidity (%)",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (ppb)",
    "SO2 (ppb)",
    "CO (ppm)",
    "Proximity to Industrial Areas (km)",
    "Population Density (people/km²): 0 < x < 50 000"
]


# -------- Safe model loader --------
def safe_load(name, file):
        model = pickle.load(open(file, "rb"))
        print(f"{name} loaded via pickle.")
        return model


# Load all models
models = {
    "XGBoost": safe_load("XGBoost", "xgboost_model.pkl"),
    "CatBoost": safe_load("CatBoost", "catboost_model.pkl"),
}


# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None

    if request.method == "POST":
        try:
            selected_model_name = request.form.get("model")
            model = models[selected_model_name]

            if model is None:
                prediction = "Model not loaded properly."
            else:
                # Collect feature inputs
                features = [float(request.form.get(f)) for f in feature_names]
                input_data = np.array([features])

                # --- Predict ---
                prediction = model.predict(input_data)[0]
                if prediction == 0:
                    prediction = "Hazardous"
                if prediction == 1:
                    prediction = "Bad"
                if prediction == 2:
                    prediction = "Moderate"
                if prediction == 3:
                    prediction = "Good"
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0].tolist()

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        feature_names=feature_names,
        models=list(models.keys()),
        prediction=prediction,

    )

# -------- Run app --------
if __name__ == "__main__":
    app.run(debug=True)
