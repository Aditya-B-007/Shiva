from __future__ import annotations
import hashlib
import logging
import sys
from typing import Iterator, Dict, Optional
from concurrent import futures

import grpc
try:
    import shiva_locomotion_pb2
    import shiva_locomotion_pb2_grpc
except ImportError:
    from locomotion import shiva_locomotion_pb2
    from locomotion import shiva_locomotion_pb2_grpc

from core.interfaces import ICognitiveSnapshot, ILocomotionTransport

# Configure structured production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ShivaLocomotion")


class CognitiveSnapshot(ICognitiveSnapshot):
    """
    Concrete implementation of ICognitiveSnapshot tracking metadata,
    node identifier, and the raw target payload state.
    """
    def __init__(self, migration_id: str, node_id: str, payload: bytes) -> None:
        self._migration_id = migration_id
        self._node_id = node_id
        self._payload = payload

    @property
    def migration_id(self) -> str:
        return self._migration_id

    @property
    def node_id(self) -> str:
        return self._node_id

    def serialise(self) -> bytes:
        return self._payload

    @classmethod
    def deserialise(cls, data: bytes) -> CognitiveSnapshot:
        # Layout depends on internal serialization requirements.
        # This assumes a primitive implementation for interface compatibility.
        import pickle
        obj = pickle.loads(data)
        return cls(obj["migration_id"], obj["node_id"], obj["payload"])


class GrpcTransport(ILocomotionTransport):
    """
    Production-grade gRPC locomotion transport layer.
    Features cryptographic payload checking, Keepalive ping management,
    and granular RPC error handling.
    """

    def __init__(
        self,
        server_cert: Optional[bytes] = None,
        private_key: Optional[bytes] = None,
        root_certificates: Optional[bytes] = None,
        max_chunk_size: int = 1024 * 1024,  # Bounded at 1MB per frame chunk
        timeout_seconds: int = 300           # 5-minute timeout window
    ) -> None:
        self.server_cert = server_cert
        self.private_key = private_key
        self.root_certificates = root_certificates
        self.max_chunk_size = max_chunk_size
        self.timeout_seconds = timeout_seconds

        # Configure hardened production channel options
        self._channel_options = [
            ('grpc.max_send_message_length', 512 * 1024 * 1024),     # Max send limit: 512MB
            ('grpc.max_receive_message_length', 512 * 1024 * 1024),  # Max receive limit: 512MB
            ('grpc.keepalive_time_ms', 30000),                       # Send pings every 30 seconds if idle
            ('grpc.keepalive_timeout_ms', 10000),                    # Wait 10 seconds for keepalive ack
            ('grpc.keepalive_permit_without_calls', 1),              # Allow keepalive pings without active calls
            ('grpc.http2.max_pings_without_data', 0),                # Unlimited pings
            ('grpc.http2.min_time_between_pings_ms', 10000),         # Min 10 seconds between pings
        ]

    def _create_credentials(self) -> Optional[grpc.ChannelCredentials]:
        if self.private_key and self.server_cert and self.root_certificates:
            try:
                return grpc.ssl_channel_credentials(
                    root_certificates=self.root_certificates,
                    private_key=self.private_key,
                    certificate_chain=self.server_cert,
                )
            except Exception as e:
                logger.error(f"Failed to compile secure mTLS credentials: {str(e)}")
                raise RuntimeError("Invalid security credentials provided for gRPC transport.")
        return None

    def send(self, snapshot: ICognitiveSnapshot, destination: str) -> str:
        """
        Packs, hashes, and streams a serialized cognitive snapshot to a target host destination.
        """
        payload = snapshot.serialise()
        total_bytes = len(payload)
        migration_id = snapshot.migration_id
        node_id = snapshot.node_id

        # Compute SHA-256 validation checksum
        sha256_hash = hashlib.sha256(payload).hexdigest()
        logger.info(f"Initiating locomotion upload for node {node_id} to {destination}. Payload size: {total_bytes} bytes. Hash: {sha256_hash}")

        creds = self._create_credentials()
        channel = (
            grpc.secure_channel(destination, creds, options=self._channel_options)
            if creds
            else grpc.insecure_channel(destination, options=self._channel_options)
        )

        with channel:
            stub = shiva_locomotion_pb2_grpc.LocomotionServiceStub(channel)

            def chunk_generator() -> Iterator[shiva_locomotion_pb2.SnapshotChunk]:
                # Stream metadata frame first
                metadata = shiva_locomotion_pb2.SnapshotMetadata(
                    migration_id=migration_id,
                    node_id=node_id,
                    total_bytes=total_bytes,
                    sha256_checksum=sha256_hash
                )
                yield shiva_locomotion_pb2.SnapshotChunk(metadata=metadata)

                # Stream raw binary frames sequentially
                for i in range(0, total_bytes, self.max_chunk_size):
                    chunk_slice = payload[i : i + self.max_chunk_size]
                    yield shiva_locomotion_pb2.SnapshotChunk(chunk_data=chunk_slice)

            try:
                # Execute blocking streaming request with an explicit deadline timeout
                response = stub.TransferSnapshot(chunk_generator(), timeout=self.timeout_seconds)
                
                if not response.success:
                    err_msg = f"Remote destination rejected locomotion. Reason: {response.message} (Code: {response.code})"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                logger.info(f"Locomotion transmission successfully finalized for migration ID: {response.migration_id}")
                return response.migration_id

            except grpc.RpcError as rpc_err:
                status_code = rpc_err.code()
                details = rpc_err.details()
                logger.error(f"gRPC locomotion transmission aborted. Code: {status_code}, Context Details: {details}")
                raise RuntimeError(f"Network transport error during locomotion [{status_code}]: {details}")


    def receive(self, migration_id: str, source: str = "") -> CognitiveSnapshot:
        """
        Fetches an operational weight snapshot from a target remote node and validates data integrity.
        """
        if not source:
            raise ValueError("A valid source endpoint address must be specified for fetch locomotion requests.")

        logger.info(f"Requesting snapshot download for migration reference: {migration_id} from {source}")
        creds = self._create_credentials()
        channel = (
            grpc.secure_channel(source, creds, options=self._channel_options)
            if creds
            else grpc.insecure_channel(source, options=self._channel_options)
        )

        with channel:
            stub = shiva_locomotion_pb2_grpc.LocomotionServiceStub(channel)
            request = shiva_locomotion_pb2.MigrationRequest(migration_id=migration_id)

            try:
                response_stream = stub.FetchSnapshot(request, timeout=self.timeout_seconds)
                
                payload_buffer = bytearray()
                expected_hash: Optional[str] = None
                expected_size: int = 0
                node_id: str = "unknown"
                metadata_parsed = False

                for frame in response_stream:
                    frame_type = frame.WhichOneof("payload")
                    
                    if not metadata_parsed:
                        if frame_type != "metadata":
                            raise RuntimeError("Protocol violation: Received data chunk before metadata block.")
                        
                        meta = frame.metadata
                        expected_hash = meta.sha256_checksum
                        expected_size = meta.total_bytes
                        node_id = meta.node_id
                        metadata_parsed = True
                        logger.info(f"Stream metadata initialized. Target size: {expected_size} bytes. Checking validation signature...")
                        continue

                    if frame_type == "chunk_data":
                        payload_buffer.extend(frame.chunk_data)

                if len(payload_buffer) == 0:
                    raise RuntimeError(f"Received empty payload stream for migration index: {migration_id}")

                if len(payload_buffer) != expected_size:
                    raise RuntimeError(f"Payload size mismatch. Received: {len(payload_buffer)}, Expected: {expected_size}")

                # Compute runtime SHA-256 verification signature
                actual_hash = hashlib.sha256(payload_buffer).hexdigest()
                if actual_hash != expected_hash:
                    logger.critical(f"CRITICAL: Cryptographic payload validation failed! In-transit modification suspected. Calculated: {actual_hash}, Expected: {expected_hash}")
                    raise RuntimeError("Locomotion aborted: SHA-256 data validation signature verification mismatch.")

                logger.info(f"Snapshot integrity confirmed. Reassembling memory structures for node: {node_id}")
                return CognitiveSnapshot(migration_id, node_id, bytes(payload_buffer))

            except grpc.RpcError as rpc_err:
                logger.error(f"Failed to fetch remote snapshot context: {rpc_err.code()} - {rpc_err.details()}")
                raise RuntimeError(f"Locomotion retrieval failed [{rpc_err.code()}]: {rpc_err.details()}")


# ---------------------------------------------------------------------------
# Server-Side Implementation Logic
# ---------------------------------------------------------------------------

class LocomotionServicer(shiva_locomotion_pb2_grpc.LocomotionServiceServicer):
    """
    Thread-safe gRPC recipient servicer node designed to parse incoming weight streams.
    """

    def __init__(self, storage_registry: Optional[Dict[str, bytes]] = None) -> None:
        # Thread-safe context mapping
        self.storage = storage_registry if storage_registry is not None else {}

    def TransferSnapshot(self, request_iterator, context) -> shiva_locomotion_pb2.MigrationStatus:
        payload_buffer = bytearray()
        migration_id: Optional[str] = None
        node_id: Optional[str] = None
        expected_hash: Optional[str] = None
        expected_size: int = 0
        metadata_received = False

        try:
            for frame in request_iterator:
                # Real-time client cancellation processing check
                if not context.is_active():
                    logger.warning("Locomotion call abandoned by upstream client.")
                    return shiva_locomotion_pb2.MigrationStatus(
                        success=False,
                        code=shiva_locomotion_pb2.MigrationStatus.StatusCode.TRANSMISSION_FAILURE,
                        message="Context cancellation triggered."
                    )

                frame_type = frame.WhichOneof("payload")

                if not metadata_received:
                    if frame_type != "metadata":
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("Metadata sequence descriptor frame must lead payload stream.")
                        return shiva_locomotion_pb2.MigrationStatus(
                            success=False,
                            code=shiva_locomotion_pb2.MigrationStatus.StatusCode.TRANSMISSION_FAILURE,
                            message="Invalid transaction sequencing."
                        )
                    
                    meta = frame.metadata
                    migration_id = meta.migration_id
                    node_id = meta.node_id
                    expected_hash = meta.sha256_checksum
                    expected_size = meta.total_bytes
                    metadata_received = True
                    continue

                if frame_type == "chunk_data":
                    payload_buffer.extend(frame.chunk_data)

            if not migration_id or len(payload_buffer) == 0:
                return shiva_locomotion_pb2.MigrationStatus(
                    success=False,
                    code=shiva_locomotion_pb2.MigrationStatus.StatusCode.TRANSMISSION_FAILURE,
                    message="Incomplete or missing transmission data segments."
                )

            # Evaluate tracking metrics and integrity constraints
            if len(payload_buffer) != expected_size:
                return shiva_locomotion_pb2.MigrationStatus(
                    migration_id=migration_id,
                    success=False,
                    code=shiva_locomotion_pb2.MigrationStatus.StatusCode.TRANSMISSION_FAILURE,
                    message=f"Size validation anomaly. Written: {len(payload_buffer)}, Bound: {expected_size}"
                )

            calculated_hash = hashlib.sha256(payload_buffer).hexdigest()
            if calculated_hash != expected_hash:
                logger.error(f"Inbound verification failure for migration {migration_id}. Expected: {expected_hash}, Calculated: {calculated_hash}")
                return shiva_locomotion_pb2.MigrationStatus(
                    migration_id=migration_id,
                    success=False,
                    code=shiva_locomotion_pb2.MigrationStatus.StatusCode.CHECKSUM_MISMATCH,
                    message="Payload signature verification failed. Bits corrupted in transit."
                )

            # Cache the confirmed snapshot payload state
            self.storage[migration_id] = bytes(payload_buffer)
            logger.info(f"Node {node_id} successfully reassembled via migration ID: {migration_id}")

            return shiva_locomotion_pb2.MigrationStatus(
                migration_id=migration_id,
                success=True,
                code=shiva_locomotion_pb2.MigrationStatus.StatusCode.OK,
                message="Snapshot reassembled and committed to local storage."
            )

        except Exception as err:
            logger.exception(f"Unhandled system error encountered in locomotion receiver channel: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal processing failure: {str(err)}")
            return shiva_locomotion_pb2.MigrationStatus(
                success=False,
                code=shiva_locomotion_pb2.MigrationStatus.StatusCode.TRANSMISSION_FAILURE,
                message=str(err)
            )

    def FetchSnapshot(self, request, context) -> Iterator[shiva_locomotion_pb2.SnapshotChunk]:
        migration_id = request.migration_id
        
        if migration_id not in self.storage:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Requested migration footprint profile {migration_id} not available.")
            return

        payload = self.storage[migration_id]
        sha256_hash = hashlib.sha256(payload).hexdigest()
        max_chunk_size = 1024 * 1024

        # Stream metadata descriptor frame
        metadata = shiva_locomotion_pb2.SnapshotMetadata(
            migration_id=migration_id,
            node_id="fetched_node_context",
            total_bytes=len(payload),
            sha256_checksum=sha256_hash
        )
        yield shiva_locomotion_pb2.SnapshotChunk(metadata=metadata)

        # Stream sequential data frames
        for i in range(0, len(payload), max_chunk_size):
            if not context.is_active():
                logger.warning(f"Downstream consumer aborted call connection while processing fetch: {migration_id}")
                break
            yield shiva_locomotion_pb2.SnapshotChunk(chunk_data=payload[i : i + max_chunk_size])


# ---------------------------------------------------------------------------
# Server Runtime Initialization Helper
# ---------------------------------------------------------------------------

def start_locomotion_server(endpoint: str, storage_registry: dict) -> grpc.Server:
    server_options = [
        ('grpc.max_send_message_length', 512 * 1024 * 1024),
        ('grpc.max_receive_message_length', 512 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
    ]
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=server_options
    )
    
    servicer = LocomotionServicer(storage_registry=storage_registry)
    shiva_locomotion_pb2_grpc.add_LocomotionServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(endpoint)
    server.start()
    logger.info(f"Shiva Locomotion gRPC Server successfully initialized on port/endpoint: {endpoint}")
    return server
