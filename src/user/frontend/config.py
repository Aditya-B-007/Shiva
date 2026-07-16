from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    backend_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: float = 10.0
    health_timeout_seconds: float = 2.0
    simulation_fallback_enabled: bool = True


def load_config() -> FrontendConfig:
    return FrontendConfig(
        backend_base_url=os.getenv("SHIVA_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/"),
        request_timeout_seconds=float(os.getenv("SHIVA_REQUEST_TIMEOUT_SECONDS", "10")),
        health_timeout_seconds=float(os.getenv("SHIVA_HEALTH_TIMEOUT_SECONDS", "2")),
        simulation_fallback_enabled=os.getenv("SHIVA_SIMULATION_FALLBACK", "1") not in {"0", "false", "False"},
    )
