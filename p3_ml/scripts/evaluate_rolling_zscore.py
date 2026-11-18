
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("rolling_zscore_v2_results.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Garder uniquement la période de test
test = df[
    df["timestamp"] >= "2026-12-04"
].copy()

y_true = test["is_anomaly"]
y_pred = test["zscore_anomaly"]

print("Nombre de mesures testées :", len(test))
print("Anomalies réelles :", y_true.sum())
print("Anomalies détectées :", y_pred.sum())

print("\nMatrice de confusion :")
print(confusion_matrix(y_true, y_pred))

print("\nPrecision :", round(precision_score(y_true, y_pred, zero_division=0), 4))
print("Recall    :", round(recall_score(y_true, y_pred, zero_division=0), 4))
print("F1-score  :", round(f1_score(y_true, y_pred, zero_division=0), 4))
