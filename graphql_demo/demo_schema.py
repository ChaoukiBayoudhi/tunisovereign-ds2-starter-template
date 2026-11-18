from datetime import datetime
import strawberry


@strawberry.type
class Cooperative:
    id: int
    name: str
    governorate: str


@strawberry.type
class Sensor:
    id: int
    name: str
    cooperative_id: int


@strawberry.type
class Telemetry:
    sensor_id: int
    soil_humidity: float   # en pourcentage (%)
    temperature: float     # en degres Celsius
    water_flow: float      # unite a valider avec P1/P3 (propose: L/min)
    timestamp: datetime


COOPERATIVES = [
    Cooperative(id=1, name="GDA Kasserine Nord", governorate="Kasserine"),
    Cooperative(id=2, name="GDA Sidi Bouzid Centre", governorate="Sidi Bouzid"),
]

SENSORS = [
    Sensor(id=1, name="Capteur parcelle 1", cooperative_id=1),
    Sensor(id=2, name="Capteur parcelle 2", cooperative_id=1),
    Sensor(id=3, name="Capteur parcelle 3", cooperative_id=2),
]

READINGS = [
    Telemetry(sensor_id=1, soil_humidity=18.5, temperature=35.2, water_flow=4.8,
            timestamp=datetime(2026, 11, 17, 10, 30)),
    Telemetry(sensor_id=2, soil_humidity=52.0, temperature=28.4, water_flow=6.1,
            timestamp=datetime(2026, 11, 17, 10, 30)),
]


@strawberry.type
class Query:
    @strawberry.field
    def cooperatives(self) -> list[Cooperative]:
        return COOPERATIVES

    @strawberry.field
    def sensors(self) -> list[Sensor]:
        return SENSORS

    @strawberry.field
    def telemetry(self) -> list[Telemetry]:
        return READINGS


schema = strawberry.Schema(query=Query)