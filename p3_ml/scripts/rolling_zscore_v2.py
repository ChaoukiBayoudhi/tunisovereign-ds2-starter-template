
import pandas as pd

# 1. Charger le dataset
df = pd.read_csv("tunichain_telemetry_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# 2. Trier les mesures
df = df.sort_values(["sensor_id", "timestamp"]).copy()

# 3. Déterminer l'état de la pompe
hour = df["timestamp"].dt.hour

df["pump_on"] = (
    ((hour >= 6) & (hour < 10)) |
    ((hour >= 18) & (hour < 20))
).astype(int)

# 4. Calculer la moyenne des 6 mesures précédentes
# du même capteur et du même état de pompe
groups = df.groupby(["sensor_id", "pump_on"])["water_flow"]

df["rolling_mean"] = groups.transform(
    lambda s: s.shift(1).rolling(6).mean()
)

# 5. Calculer l'écart-type
df["rolling_std"] = groups.transform(
    lambda s: s.shift(1).rolling(6).std(ddof=0)
)

# 6. Calculer le Z-Score
df["z_score"] = (
    (df["water_flow"] - df["rolling_mean"]) /
    df["rolling_std"].replace(0, float("nan"))
)

# 7. Détecter les anomalies
df["zscore_anomaly"] = (
    df["z_score"].abs() > 3
).fillna(False).astype(int)

# 8. Afficher les résultats
print("Nombre total de mesures :", len(df))
print("Anomalies détectées :", df["zscore_anomaly"].sum())

# 9. Sauvegarder
df.to_csv("rolling_zscore_v2_results.csv", index=False)

print("Résultats sauvegardés.")
