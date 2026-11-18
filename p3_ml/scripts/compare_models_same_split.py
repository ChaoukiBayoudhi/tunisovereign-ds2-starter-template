
import pandas as pd
from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Chargement et prÃ©paration des donnÃ©es
ROOT = Path(__file__).resolve().parent.parent`nDATA_DIR = ROOT / "data"`nRESULTS_DIR = ROOT / "results"`nRESULTS_DIR.mkdir(exist_ok=True)`ndf = pd.read_csv(DATA_DIR / "tunichain_telemetry_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

numeric_columns = [
    "soil_humidity",
    "temperature",
    "water_flow",
    "is_anomaly"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_columns + ["timestamp"])
df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

# 2. CrÃ©ation de la feature pump_on
hour = df["timestamp"].dt.hour

df["pump_on"] = (
    ((hour >= 6) & (hour < 10)) |
    ((hour >= 18) & (hour < 20))
).astype(int)

# 3. SÃ©paration chronologique identique pour les deux modÃ¨les
df = df.sort_values("timestamp").reset_index(drop=True)

unique_dates = sorted(df["timestamp"].dt.date.unique())

split_index = int(len(unique_dates) * 0.80)
cutoff_date = unique_dates[split_index]

train_df = df[df["timestamp"].dt.date < cutoff_date].copy()
test_df = df[df["timestamp"].dt.date >= cutoff_date].copy()

y_train = train_df["is_anomaly"].astype(int)
y_test = test_df["is_anomaly"].astype(int)

print("===== DONNEES COMMUNES =====")
print("Date limite :", cutoff_date)
print("Train :", len(train_df))
print("Test :", len(test_df))

# 4. DÃ©finition des deux groupes de features
features_A = [
    "soil_humidity",
    "temperature",
    "water_flow"
]

features_B = [
    "soil_humidity",
    "temperature",
    "water_flow",
    "pump_on"
]

# MÃªme contamination et mÃªmes paramÃ¨tres
contamination_rate = float(y_train.mean())

def evaluate_model(name, features, model_filename):
    X_train = train_df[features]
    X_test = test_df[features]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    predictions = model.predict(X_test)
    y_pred = (predictions == -1).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n===== {name} =====")
    print("Features :", features)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport dÃ©taillÃ© :")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Analyse par type d'anomalie
    results = test_df[["sensor_id", "timestamp", "anomaly_type"]].copy()
    results["is_anomaly"] = y_test.values
    results["predicted_anomaly"] = y_pred

    anomalies = results[results["is_anomaly"] == 1]

    summary = anomalies.groupby("anomaly_type").agg(
        total=("is_anomaly", "count"),
        detected=("predicted_anomaly", "sum")
    )

    summary["missed"] = summary["total"] - summary["detected"]
    summary["recall_percent"] = (
        summary["detected"] / summary["total"] * 100
    ).round(2)

    print("\nDÃ©tection par type :")
    print(summary)

    joblib.dump(model, RESULTS_DIR / model_filename)
    results.to_csv(RESULTS_DIR / f"results_{name}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"analysis_{name}.csv")

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# 5. EntraÃ®nement et comparaison
result_A = evaluate_model(
    "Model_A_3_features",
    features_A,
    "model_A_3_features.joblib"
)

result_B = evaluate_model(
    "Model_B_4_features",
    features_B,
    "model_B_4_features.joblib"
)

# 6. Tableau comparatif final
comparison = pd.DataFrame([result_A, result_B])

print("\n===== COMPARAISON FINALE =====")
print(comparison.to_string(index=False))

comparison.to_csv(RESULTS_DIR / "model_comparison_same_split.csv", index=False)

print("\nComparaison sauvegardÃ©e.")
print("PIPELINE COMPARISON FINISHED")

