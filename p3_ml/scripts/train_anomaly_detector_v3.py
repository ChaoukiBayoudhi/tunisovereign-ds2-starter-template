
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Charger le dataset
df = pd.read_csv("tunichain_telemetry_dataset.csv")

print("Dataset initial :", df.shape)

# 2. Nettoyer les données
numeric_columns = ["soil_humidity", "temperature", "water_flow"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_columns + ["is_anomaly"])
df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

# 3. Préparer les dates
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)

# 4. Ajouter l'état prévu de la pompe
hour = df["timestamp"].dt.hour

df["pump_on"] = (
    ((hour >= 6) & (hour < 10)) |
    ((hour >= 18) & (hour < 20))
).astype(int)

# 5. Séparation temporelle 80% / 20%
unique_dates = sorted(df["timestamp"].unique())

split_index = int(len(unique_dates) * 0.80)
cutoff_date = unique_dates[split_index]

train_df = df[df["timestamp"] < cutoff_date].copy()
test_df = df[df["timestamp"] >= cutoff_date].copy()

features = [
    "soil_humidity",
    "temperature",
    "water_flow",
    "pump_on"
]

X_train = train_df[features]
y_train = train_df["is_anomaly"].astype(int)

X_test = test_df[features]
y_test = test_df["is_anomaly"].astype(int)

print("\n===== TEMPORAL SPLIT =====")
print("Date limite :", cutoff_date)
print("Entraînement :", X_train.shape)
print("Test :", X_test.shape)

print("\nPériode entraînement :")
print(train_df["timestamp"].min(), "->", train_df["timestamp"].max())

print("\nPériode test :")
print(test_df["timestamp"].min(), "->", test_df["timestamp"].max())

# Vérifier la présence d'anomalies dans chaque période
print("\nAnomalies train :", y_train.sum())
print("Anomalies test :", y_test.sum())

# 6. Définir la contamination à partir du train uniquement
contamination = float(y_train.mean())

# 7. Entraîner IsolationForest
model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

# 8. Prédictions
predictions = model.predict(X_test)
y_pred = (predictions == -1).astype(int)

# 9. Évaluation
print("\n===== EVALUATION MODELE V3 =====")

print("Precision :", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))

print("Recall :", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))

print("F1-Score :", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Sauvegarder le modèle
joblib.dump(model, "isolation_forest_model_v3.joblib")

# 11. Sauvegarder les résultats
results = test_df[[
    "sensor_id",
    "timestamp",
    "anomaly_type",
    "is_anomaly",
    "pump_on"
]].copy()

results["predicted_anomaly"] = y_pred

results.to_csv("anomaly_detection_results_v3.csv", index=False)

print("\nModèle sauvegardé : isolation_forest_model_v3.joblib")
print("Résultats sauvegardés : anomaly_detection_results_v3.csv")
print("\nMACHINE LEARNING PIPELINE V3 FINISHED")
