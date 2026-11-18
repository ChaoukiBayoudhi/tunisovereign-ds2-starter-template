
import pandas as pd

df = pd.read_csv("anomaly_detection_results_v3.csv")

anomalies = df[df["is_anomaly"] == 1].copy()

summary = anomalies.groupby("anomaly_type").agg(
    total=("is_anomaly", "count"),
    detected=("predicted_anomaly", "sum")
)

summary["missed"] = summary["total"] - summary["detected"]

summary["recall_percent"] = (
    summary["detected"] / summary["total"] * 100
).round(2)

print("\n===== DETECTION BY ANOMALY TYPE V3 =====")
print(summary)

summary.to_csv("anomaly_type_analysis_v3.csv")

print("\nRésultats sauvegardés dans anomaly_type_analysis_v3.csv")
