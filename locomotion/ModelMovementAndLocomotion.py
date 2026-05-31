from __future__ import annotations
import hashlib
import hmac
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib import request as urllib_request
import torch
import torch.nn as nn
from core.interfaces import ICognitiveSnapshot, ILocomotionTransport

SNAPSHOT_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# CognitiveSnapshot
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMetadata:
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)
    payload_bytes: int = 0
    hmac_hex: str = ""          # populated after serialisation
    migration_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CognitiveSnapshot(ICognitiveSnapshot):
    """
    A fully self-contained, rehydratable representation of a Shiva node.

    Serialisation format (big-endian, no external dependencies beyond torch):

        [ 4 bytes  ] header_length (uint32)
        [ N bytes  ] JSON header (SnapshotMetadata)
        [ M bytes  ] torch.save() blob — the state dict bundle

    The state dict bundle contains:
        {
          "model_state":     OrderedDict from policy.state_dict(),
          "memory_bank":     list of episode dicts (detached tensors),
          "emotional_state": { "mood": str, "homeostasis": Tensor[4] },
          "identity_token":  Tensor[1, 1, D],
        }

    HMAC-SHA256 is computed over the JSON header + tensor blob concatenated.
    The hex digest is stored inside the header's `hmac_hex` field.

    Note on `weights_only`:
        torch.load with weights_only=True restricts deserialisation to
        safe tensor types only.  This is the secure default.  Set
        `weights_only=False` only in a fully trusted environment.
    """

    def __init__(
        self,
        metadata: SnapshotMetadata,
        state_bundle: Dict[str, Any],
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        self.metadata = metadata
        self._state_bundle = state_bundle
        self._hmac_secret = hmac_secret or b"shiva-dev-secret"

    # ------------------------------------------------------------------
    # ICognitiveSnapshot
    # ------------------------------------------------------------------

    def serialise(self) -> bytes:
        """
        Pack cognitive state into a signed, versioned byte payload.

        Steps:
          1. torch.save the state bundle to an in-memory buffer.
          2. Update metadata with the buffer length.
          3. Compute HMAC-SHA256 over (header_json + tensor_blob).
          4. Pack: [uint32 header_len][header_json][tensor_blob].
        """
        # Serialise the tensor bundle.
        tensor_buf = io.BytesIO()
        torch.save(self._state_bundle, tensor_buf)
        tensor_bytes = tensor_buf.getvalue()

        # Compute HMAC over tensor payload first (header not yet final).
        self.metadata.payload_bytes = len(tensor_bytes)
        self.metadata.hmac_hex = ""   # zero out before signing

        header_json = json.dumps(
            {
                "schema_version": self.metadata.schema_version,
                "node_id": self.metadata.node_id,
                "timestamp": self.metadata.timestamp,
                "payload_bytes": self.metadata.payload_bytes,
                "migration_id": self.metadata.migration_id,
            }
        ).encode("utf-8")

        sig = hmac.new(
            self._hmac_secret,
            header_json + tensor_bytes,
            hashlib.sha256,
        ).hexdigest()
        self.metadata.hmac_hex = sig

        # Re-serialise header with HMAC included.
        header_json = json.dumps(
            {
                "schema_version": self.metadata.schema_version,
                "node_id": self.metadata.node_id,
                "timestamp": self.metadata.timestamp,
                "payload_bytes": self.metadata.payload_bytes,
                "hmac_hex": self.metadata.hmac_hex,
                "migration_id": self.metadata.migration_id,
            }
        ).encode("utf-8")

        header_len = len(header_json).to_bytes(4, "big")
        return header_len + header_json + tensor_bytes

    @classmethod
    def deserialise(
        cls,
        payload: bytes,
        hmac_secret: Optional[bytes] = None,
    ) -> "CognitiveSnapshot":
        """
        Reconstruct a CognitiveSnapshot from a byte payload.

        Raises ValueError on HMAC mismatch or schema version incompatibility.
        """
        secret = hmac_secret or b"shiva-dev-secret"

        # Unpack.
        header_len = int.from_bytes(payload[:4], "big")
        header_json = payload[4 : 4 + header_len]
        tensor_bytes = payload[4 + header_len :]

        meta_dict = json.loads(header_json.decode("utf-8"))

        # Schema version check.
        if meta_dict.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Snapshot schema version mismatch: "
                f"expected {SNAPSHOT_SCHEMA_VERSION}, "
                f"got {meta_dict.get('schema_version')}."
            )

        # HMAC verification.
        received_sig = meta_dict.pop("hmac_hex", "")
        clean_header = json.dumps(
            {k: v for k, v in meta_dict.items() if k != "hmac_hex"}
        ).encode("utf-8")
        expected_sig = hmac.new(
            secret, clean_header + tensor_bytes, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(received_sig, expected_sig):
            raise ValueError(
                "CognitiveSnapshot HMAC verification failed. "
                "Payload may be corrupted or tampered."
            )

        # Restore metadata.
        metadata = SnapshotMetadata(
            schema_version=meta_dict["schema_version"],
            node_id=meta_dict.get("node_id", ""),
            timestamp=meta_dict.get("timestamp", 0.0),
            payload_bytes=meta_dict.get("payload_bytes", 0),
            hmac_hex=received_sig,
            migration_id=meta_dict.get("migration_id", ""),
        )

        # Restore tensor bundle.
        buf = io.BytesIO(tensor_bytes)
        state_bundle = torch.load(buf, weights_only=True, map_location="cpu")

        return cls(metadata, state_bundle, secret)

    # ------------------------------------------------------------------
    # Convenience constructors and restoration
    # ------------------------------------------------------------------

    @classmethod
    def capture(
        cls,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        node_id: str = "",
        hmac_secret: Optional[bytes] = None,
    ) -> "CognitiveSnapshot":
        """
        Capture a snapshot from live modules.

        Accesses:
            policy.state_dict()                            — model weights
            episodic_memory._bank                          — episode deque
            emotional_core._homeostasis.vector             — homeostasis state
            emotional_core._mood.name                      — current mood
            episodic_memory.self_token                     — identity token
        """
        # Serialise the episode bank: each episode has a 'states' tensor
        # and a 'significance' float.
        memory_bank = [
            {
                "states": ep["states"].cpu(),
                "significance": float(ep["significance"]),
            }
            for ep in list(episodic_memory._bank)  # type: ignore[attr-defined]
        ]

        bundle = {
            "model_state": {
                k: v.cpu() for k, v in policy.state_dict().items()
            },
            "memory_bank": memory_bank,
            "emotional_state": {
                "mood": emotional_core._mood.name,  # type: ignore[attr-defined]
                "homeostasis": emotional_core._homeostasis.vector.detach().cpu(),  # type: ignore[attr-defined]
            },
            "identity_token": episodic_memory.self_token.detach().cpu(),  # type: ignore[attr-defined]
        }

        metadata = SnapshotMetadata(node_id=node_id)
        return cls(metadata, bundle, hmac_secret)

    def restore(
        self,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        device: str = "cpu",
    ) -> None:
        """
        Restore the captured cognitive state into live modules in-place.

        Accesses the same attributes as `capture`; modules must be
        structurally compatible with the snapshot.
        """
        bundle = self._state_bundle
        dev = torch.device(device)

        # Restore model weights.
        policy.load_state_dict(
            {k: v.to(dev) for k, v in bundle["model_state"].items()},
            strict=False,
        )

        # Restore episodic memory bank.
        from collections import deque
        episodic_memory._bank = deque(  # type: ignore[attr-defined]
            [
                {
                    "states": ep["states"].to(dev),
                    "significance": ep["significance"],
                }
                for ep in bundle["memory_bank"]
            ],
            maxlen=episodic_memory.capacity,  # type: ignore[attr-defined]
        )

        # Restore identity token.
        with torch.no_grad():
            episodic_memory.self_token.copy_(  # type: ignore[attr-defined]
                bundle["identity_token"].to(dev)
            )

        # Restore emotional state.
        emo = bundle["emotional_state"]
        with torch.no_grad():
            emotional_core._homeostasis._state.copy_(  # type: ignore[attr-defined]
                emo["homeostasis"].to(dev)
            )
        emotional_core._mood.transition(emo["mood"], "Restored from snapshot")  # type: ignore[attr-defined]

    @property
    def migration_id(self) -> str:
        return self.metadata.migration_id

    @property
    def node_id(self) -> str:
        return self.metadata.node_id


# ---------------------------------------------------------------------------
# HTTP transport (development / local use)
# ---------------------------------------------------------------------------

class HttpTransport(ILocomotionTransport):
    """
    Transmits cognitive snapshots over plain HTTP.

    Protocol:
      • POST  /migrate          — send snapshot bytes; server returns migration_id
      • GET   /migrate/{id}     — poll for snapshot bytes; 202 = pending, 200 = ready

    This transport is intentionally simple.  For production, use GrpcTransport
    (mutual TLS, streaming, back-pressure) or replace entirely.

    Security note: plain HTTP only for local development or a trusted
    internal network.  Always terminate TLS at the load balancer in production.
    """

    _CONTENT_TYPE = "application/octet-stream"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout = timeout_seconds

    def send(self, snapshot: ICognitiveSnapshot, destination: str) -> str:
        """
        POST the snapshot payload to `destination/migrate`.

        Args:
            snapshot:    CognitiveSnapshot to transmit.
            destination: Base URL, e.g. 'http://192.168.1.42:8080'.

        Returns:
            migration_id (str) assigned by the receiving server.
        """
        payload = snapshot.serialise()
        url = destination.rstrip("/") + "/migrate"

        req = urllib_request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": self._CONTENT_TYPE},
        )
        with urllib_request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
            return body["migration_id"]

    def receive(self, migration_id: str, source: str = "") -> "CognitiveSnapshot":
        """
        Poll `source/migrate/{migration_id}` until the snapshot is available.

        Args:
            migration_id: ID returned by a prior `send` call.
            source:       Base URL of the sending node.

        Returns:
            Deserialised CognitiveSnapshot.
        """
        url = f"{source.rstrip('/')}/migrate/{migration_id}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            req = urllib_request.Request(url, method="GET")
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                if resp.status == 200:
                    return CognitiveSnapshot.deserialise(resp.read())
                elif resp.status == 202:
                    time.sleep(0.5)   # pending — retry
                else:
                    raise RuntimeError(
                        f"Unexpected status {resp.status} from {url}"
                    )
        raise TimeoutError(
            f"Migration {migration_id} did not complete within {self.timeout}s."
        )


# ---------------------------------------------------------------------------
# gRPC transport stub (production-grade)
# ---------------------------------------------------------------------------

class GrpcTransport(ILocomotionTransport):
    """
    gRPC-based transport for production deployments.

    Requires the `grpcio` package and a generated stub from the
    `shiva_locomotion.proto` service definition:

        service LocomotionService {
          rpc Migrate (MigrateRequest) returns (MigrateResponse);
          rpc Fetch   (FetchRequest)   returns (stream FetchResponse);
        }

    This class is a structural stub.  Implement `send` and `receive`
    once the protobuf stubs are generated.

    Advantages over HttpTransport:
      • Bidirectional streaming — large snapshots chunked automatically.
      • Mutual TLS built into the channel configuration.
      • Back-pressure and flow control at the protocol level.
      • ~10× lower latency than HTTP/JSON for binary payloads.
    """

    def __init__(self, channel_credentials=None) -> None:
        self._credentials = channel_credentials
        # Stub populated by `_connect(address)` before use.
        self._stub = None

    def send(self, snapshot: ICognitiveSnapshot, destination: str) -> str:
        raise NotImplementedError(
            "GrpcTransport.send requires grpcio and a generated proto stub. "
            "Run `python -m grpc_tools.protoc` on shiva_locomotion.proto first."
        )

    def receive(self, migration_id: str) -> "CognitiveSnapshot":
        raise NotImplementedError(
            "GrpcTransport.receive requires grpcio and a generated proto stub."
        )


# ---------------------------------------------------------------------------
# LocomotionEngine: orchestrates capture → send → receive → restore
# ---------------------------------------------------------------------------

class LocomotionEngine:
    """
    High-level API for migrating a Shiva node to a remote host.

    Usage:
        engine = LocomotionEngine(
            transport=HttpTransport(),
            hmac_secret=b"my-shared-secret",
        )

        # Outbound (sender side):
        migration_id = engine.migrate_out(
            policy=policy,
            episodic_memory=episodic_memory,
            emotional_core=emotional_core,
            destination="http://10.0.0.5:8080",
            node_id="shiva-alpha",
        )

        # Inbound (receiver side — called on the remote host):
        engine.migrate_in(
            migration_id=migration_id,
            source="http://10.0.0.4:8080",
            policy=policy,
            episodic_memory=episodic_memory,
            emotional_core=emotional_core,
        )
    """

    def __init__(
        self,
        transport: ILocomotionTransport,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        self.transport = transport
        self.hmac_secret = hmac_secret or b"shiva-dev-secret"

    def migrate_out(
        self,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        destination: str,
        node_id: str = "",
    ) -> str:
        """
        Capture the current cognitive state and send it to `destination`.

        Returns the migration_id for the receiver to use when calling
        `migrate_in`.
        """
        snapshot = CognitiveSnapshot.capture(
            policy=policy,
            episodic_memory=episodic_memory,
            emotional_core=emotional_core,
            node_id=node_id,
            hmac_secret=self.hmac_secret,
        )
        migration_id = self.transport.send(snapshot, destination)
        return migration_id

    def migrate_in(
        self,
        migration_id: str,
        source: str,
        policy: nn.Module,
        episodic_memory: nn.Module,
        emotional_core: nn.Module,
        device: str = "cpu",
    ) -> None:
        """
        Receive a snapshot from `source` and restore it into the provided
        modules in-place.
        """
        snapshot = self.transport.receive(migration_id, source)
        if not isinstance(snapshot, CognitiveSnapshot):
            raise TypeError(
                f"Transport returned {type(snapshot).__name__}, "
                "expected CognitiveSnapshot."
            )
        snapshot.restore(
            policy=policy,
            episodic_memory=episodic_memory,
            emotional_core=emotional_core,
            device=device,
        )
