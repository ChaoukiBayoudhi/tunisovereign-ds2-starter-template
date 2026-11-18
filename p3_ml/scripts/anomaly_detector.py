import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "results" / "model_B_4_features.joblib"

model = joblib.load(MODEL_PATH)

def detect_anomaly(soil_humidity, temperature, water_flow, hour):
    pump_on = int((6 <= hour < 10) or (18 <= hour < 20))

    data = pd.DataFrame([{
        "soil_humidity": soil_humidity,
        "temperature": temperature,
        "water_flow": water_flow,
        "pump_on": pump_on
    }])

    prediction = model.predict(data)[0]
    anomaly_score = model.decision_function(data)[0]

    return {
        "is_anomaly": int(prediction == -1),
        "anomaly_score": round(float(-anomaly_score), 4)
    }

if __name__ == "__main__":
    result = detect_anomaly(59.51, 10.61, 2.75, 8)
    print("Detection result:")
    print(result)
