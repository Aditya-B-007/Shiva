from __future__ import annotations

from typing import Any, Dict, List

import requests

from src.contracts.api import BLOCK_SCHEMAS_PATH, HEALTH_PATH, WORKFLOW_PATH
from src.user.frontend.config import FrontendConfig


class ShivaApiClient:
    def __init__(self, config: FrontendConfig) -> None:
        self._config = config

    @property
    def base_url(self) -> str:
        return self._config.backend_base_url

    def health(self) -> Dict[str, Any]:
        response = requests.get(
            self._url(HEALTH_PATH),
            timeout=self._config.health_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def block_schemas(self) -> List[Dict[str, Any]]:
        response = requests.get(
            self._url(BLOCK_SCHEMAS_PATH),
            timeout=self._config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def execute_workflow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._url(WORKFLOW_PATH),
            json=payload,
            timeout=self._config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"
