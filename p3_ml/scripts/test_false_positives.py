import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tunichain_telemetry_dataset.csv"
MODEL_PATH = ROOT / "results" / "model_B_4_features.joblib"

df = pd.read_csv(DATA_PATH)

df = df[df["is_anomaly"] == 0].sample(
    n=1000, random_state=42
).copy()

df["hour"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour

df["pump_on"] = (
    ((df["hour"] >= 6) & (df["hour"] < 10)) |
    ((df["hour"] >= 18) & (df["hour"] < 20))
).astype(int)

features = [
    "soil_humidity", "temperature", "water_flow", "pump_on"
]

model = joblib.load(MODEL_PATH)
predictions = model.predict(df[features])

false_positives = (predictions == -1).sum()

print("Nombre de mesures normales testees :", len(df))
print("Fausses alertes :", false_positives)
print("Taux de fausses alertes :", round(false_positives / len(df) * 100, 2), "%")
