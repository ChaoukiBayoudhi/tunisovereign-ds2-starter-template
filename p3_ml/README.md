# P3 — Data Preparation & Machine Learning

## 1. Objectif

La partie P3 prépare les données de télémétrie agricole et développe un premier détecteur d'anomalies pour le projet TuniChain-Sovereign.

## 2. Structure

- `data/` : dataset de télémétrie et exemple JSON gRPC.
- `scripts/` : génération, entraînement, détection et évaluation.
- `results/` : modèle entraîné et résultats expérimentaux.

## 3. Données utilisées

Le dataset est synthétique : il ne provient pas de capteurs agricoles réels.

- 151 200 observations.
- 35 capteurs.
- Période : du 10 novembre au 9 décembre 2026.
- Fréquence : une mesure toutes les 10 minutes.

Champs principaux :

- `sensor_id`
- `soil_humidity`
- `temperature`
- `water_flow`
- `timestamp`
- `governorate`
- `is_anomaly`
- `anomaly_type`

Les données simulées servent à développer et tester le pipeline. Les performances obtenues ne garantissent pas les mêmes résultats sur des capteurs réels.

## 4. Prérequis et installation

Prérequis :

- Python 3.13
- pandas
- numpy
- scikit-learn
- joblib

Installation des bibliothèques :

```bash
python -m pip install pandas numpy scikit-learn joblib
```

## 5. Exécution

Lancer le détecteur d'anomalies :

```bash
python p3_ml/scripts/anomaly_detector.py
```

Tester les fausses alertes sur un échantillon de 1 000 mesures normales :

```bash
python p3_ml/scripts/test_false_positives.py
```

Le modèle utilisé est `results/model_B_4_features.joblib`.

## 6. Résultats du modèle

Comparaison chronologique sur le même jeu de test :

| Métrique | Model A | Model B |
|---|---:|---:|
| Precision | 76.13 % | 89.14 % |
| Recall | 82.67 % | 93.67 % |
| F1-score | 79.26 % | 91.34 % |

Model A utilise trois caractéristiques. Model B utilise quatre caractéristiques, dont `pump_on`, estimé à partir d'un horaire supposé.

Le modèle B obtient les meilleurs résultats globaux sur cette évaluation.

## 7. Rolling Z-Score

Résultats de la version expérimentale V2 :

- Precision : 8.2 %
- Recall : 12.42 %
- F1-score : 9.87 %

Cette méthode nécessite encore des améliorations.

## 8. Limites

- Les données sont synthétiques.
- Le calendrier `pump_on` reste une hypothèse à confirmer avec P2.
- Les performances doivent être validées sur des données réelles.
- Le Rolling Z-Score actuel génère trop de fausses alertes.
- Le score d'anomalie du modèle n'est pas une probabilité.
