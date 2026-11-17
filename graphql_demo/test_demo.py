import json
from demo_schema import schema

QUERY_1 = """
query {
  sensors {
    id
    name
  }
}
"""

QUERY_2 = """
query {
  telemetry {
    sensorId
    soilHumidity
    temperature
    waterFlow
  }
}
"""

for title, query in [("Requete 1: capteurs", QUERY_1), ("Requete 2: telemetrie", QUERY_2)]:
    result = schema.execute_sync(query)
    print(f"--- {title} ---")
    print("Erreurs:", result.errors)
    print(json.dumps(result.data, indent=2, ensure_ascii=False))