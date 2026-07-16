from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger("shiva.orchestrator.server")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    server_libraries_available = True
except ImportError:
    server_libraries_available = False
    logger.warning("FastAPI or uvicorn is not installed. Standalone server mode will not be available.")

from src.orchestrator.core import ShivaOrchestrator
from src.contracts import block_schemas
from src.orchestrator.schema import Workflow

def create_app(orchestrator: Optional[ShivaOrchestrator] = None) -> FastAPI:
    """
    Creates and configures the FastAPI application for hosting the Shiva Orchestrator.
    """
    if not server_libraries_available:
        raise RuntimeError("FastAPI and uvicorn are required to launch the standalone local server.")

    app = FastAPI(
        title="shiva.ai API Server",
        description="Local loopback execution server for desktop, mobile, and custom interfaces.",
        version="1.0.0"
    )

    # Enable CORS for frontend applications (like web interfaces, Tauri, Electron)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Use shared orchestrator or instantiate a new one
    orch = orchestrator or ShivaOrchestrator()

    @app.get("/health")
    def health():
        return {
            "status": "online",
            "cognitive_core": "online" if orch.cognitive_initialized else "offline"
        }

    @app.get("/api/devices")
    def get_devices():
        """Lists all registered perception devices and their expected parameter definitions."""
        try:
            return orch.get_available_devices()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/block-schemas")
    def get_block_schemas():
        """Lists frontend-safe workflow block schemas."""
        return [schema.to_json() for schema in block_schemas()]

    @app.post("/api/workflow")
    def execute_workflow(payload: dict):
        """Executes a complete drag-and-drop workflow layout synchronously."""
        try:
            workflow = Workflow.from_dict(payload)
            result = orch.execute_workflow(workflow.to_json())
            if result.get("metadata", {}).get("error"):
                raise HTTPException(status_code=400, detail=result.get("text"))
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/api/workflow/ws")
    async def execute_workflow_ws(websocket: WebSocket):
        """Handles streaming real-time status updates and execution results."""
        await websocket.accept()
        try:
            while True:
                # Receive workflow configuration from the frontend client
                data = await websocket.receive_json()
                
                # Stream status updates back to the UI
                await websocket.send_json({
                    "status": "starting",
                    "message": "Initiating workflow execution sequence."
                })
                
                await websocket.send_json({
                    "status": "capturing",
                    "message": "Triggering hardware perception sensors."
                })
                
                await websocket.send_json({
                    "status": "reasoning",
                    "message": "Orchestrating cortical columns (Swarm reasoning cycles active)."
                })
                
                # Execute core loop
                workflow = Workflow.from_dict(data)
                result = orch.execute_workflow(workflow.to_json())
                
                await websocket.send_json({
                    "status": "completed",
                    "message": "Swarm successfully reached decision state.",
                    "result": result
                })
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        except Exception as e:
            logger.error(f"WebSocket execution error: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Execution failed: {str(e)}"
                })
            except Exception:
                pass

    return app

def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launches the uvicorn ASGI server hosting Shiva."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)
