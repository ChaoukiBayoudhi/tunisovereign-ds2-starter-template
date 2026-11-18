from datetime import datetime, timezone
import grpc

import tunisovereign_pb2 as pb2
import tunisovereign_pb2_grpc as pb2_grpc


def generate_packets():
    # L'humidité du sol chute: 45 -> 40.2 -> 30.1 (sous le seuil de 35)
    for value in [45.0, 40.2, 30.1]:
        yield pb2.TelemetryPacket(
            node_code="NODE-001",
            governorate="Kasserine",
            metric_type="MOISTURE",
            value=value,
            iso_timestamp=datetime.now(timezone.utc).isoformat(),
            signature_hmac="demo",
        )


def run():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = pb2_grpc.TelemetryIngestServiceStub(channel)

        print("--- Envoi de 3 mesures (StreamTelemetry) ---")
        ack = stub.StreamTelemetry(generate_packets())
        print("Reponse du serveur:")
        print(ack)

        print("--- Demande d'audit urgent (TriggerUrgentAudit) ---")
        audit = stub.TriggerUrgentAudit(
            pb2.AuditRequest(node_code="NODE-001", reason="Fuite suspectee")
        )
        print("Reponse du serveur:")
        print(audit)


if __name__ == "__main__":
    run()