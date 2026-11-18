from concurrent import futures
import grpc

import tunisovereign_pb2 as pb2
import tunisovereign_pb2_grpc as pb2_grpc


class TelemetryIngestServicer(pb2_grpc.TelemetryIngestServiceServicer):

    def StreamTelemetry(self, request_iterator, context):
        accepted = 0
        anomaly = False
        thread_id = ""
        for packet in request_iterator:
            accepted += 1
            print(f"Recu: {packet.node_code} | {packet.governorate} | "
                f"{packet.metric_type} = {packet.value} | {packet.iso_timestamp}")
            if packet.metric_type == "MOISTURE" and packet.value < 35.0:
                anomaly = True
                thread_id = f"demo-{packet.node_code}-{accepted}"
        return pb2.IngestAck(
            accepted_count=accepted,
            anomaly_triggered=anomaly,
            agent_thread_id=thread_id,
        )

    def TriggerUrgentAudit(self, request, context):
        print(f"Audit demande pour {request.node_code}: {request.reason}")
        return pb2.AuditResponse(status="DEMO_RECEIVED", tx_hash="")


def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_TelemetryIngestServiceServicer_to_server(TelemetryIngestServicer(), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    print(f"[Demo] Serveur gRPC en ecoute sur 127.0.0.1:{port} (Ctrl+C pour arreter)")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()