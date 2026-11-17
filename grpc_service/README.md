# gRPC + GraphQL (démo) : travail de P2

Prototype indépendant de Django et de la base de données. Les données sont fictives.
La branche est `feature/graphql-grpc`.

## Prérequis

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install strawberry-graphql grpcio grpcio-tools protobuf
```

## 1. gRPC

Le contrat est dans `protos/tunisovereign.proto`.

### Générer les fichiers Python

```powershell
.\.venv\Scripts\python.exe -m grpc_tools.protoc -I=protos --python_out=grpc_service --grpc_python_out=grpc_service protos/tunisovereign.proto
```

Les fichiers `*_pb2.py` et `*_pb2_grpc.py` sont générés : ne pas les modifier à la main.

### Lancer le serveur de démonstration (terminal 1)

```powershell
.\.venv\Scripts\python.exe grpc_service\server_demo.py
```

### Lancer le client de test (terminal 2)

```powershell
.\.venv\Scripts\python.exe grpc_service\client_test.py
```

Résultat attendu côté client : `accepted_count: 3`, `anomaly_triggered: true`
et `agent_thread_id: "demo-NODE-001-3"`. Côté serveur : 3 lignes `Recu: ...`
et une ligne `Audit demande ...`.

### Service et méthodes

| Méthode | Type | Requête | Réponse |
|---|---|---|---|
| `StreamTelemetry` | client-streaming | `TelemetryPacket` (plusieurs) | `IngestAck` |
| `TriggerUrgentAudit` | unaire | `AuditRequest` | `AuditResponse` |

Note : `grpc_service/server.py` est le serveur d'origine. Il n'est pas encore
relié au service gRPC et dépend de l'agent LangGraph (P4). `server_demo.py` sert
uniquement à tester la communication.

## 2. GraphQL (démo avec données fictives)

Fichiers : `graphql_demo/demo_schema.py` et `graphql_demo/test_demo.py`.

```powershell
.\.venv\Scripts\python.exe graphql_demo\test_demo.py
```

Exemple de requête :

```graphql
query {
  sensors {
    id
    name
  }
}
```

Le schéma n'est pas encore connecté aux modèles Django (en attente de la
validation de P1). Le fichier existant `backend/tunisovereign_core/schema_graphql.py`
n'a pas été modifié.

## 3. Champs et unités (à valider avec P1 et P3)

### Message gRPC `TelemetryPacket`

| Champ | Type | Description |
|---|---|---|
| `node_code` | string | Identifiant du capteur, ex. `"NODE-001"` |
| `governorate` | string | Gouvernorat, ex. `"Kasserine"` |
| `metric_type` | string | `MOISTURE`, `FLOW`, `TEMP` ou `STOCK` |
| `value` | double | Valeur mesurée (unité selon `metric_type`) |
| `iso_timestamp` | string | Date et heure en UTC, format ISO 8601 |
| `signature_hmac` | string | Signature du paquet (valeur factice dans la démo) |

### Unités proposées (en attente de validation)

| Mesure | Unité |
|---|---|
| Humidité du sol (`MOISTURE`) | pourcentage (%) |
| Température (`TEMP`) | degrés Celsius (°C) |
| Débit (`FLOW`) | litres par minute (L/min) |
| Timestamp | UTC, ISO 8601 |

### Types GraphQL de démonstration

| Type | Champs |
|---|---|
| `Cooperative` | `id`, `name`, `governorate` |
| `Sensor` | `id`, `name`, `cooperative_id` |
| `Telemetry` | `sensor_id`, `soil_humidity`, `temperature`, `water_flow`, `timestamp` |

## 4. Points à décider avec P1 et P3

- Identifiant du capteur : texte (`node_code`) partout, ou entier ?
- Une valeur par paquet (format actuel du `.proto`) ou plusieurs mesures ensemble ?
- Comment signaler une mesure manquante ?
- Comment brancher l'agent de P4 dans `server.py` ?