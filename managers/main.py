from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from pathlib import Path
from .capability import capability_registry
from .runtime import runtime_manager
from .attachment import attachment_manager
from .merge import merge_manager
from .execution import execution_manager, ExecutionDirective
import os

app = FastAPI(title="Shiva AGI Local Gateway", version="1.0.0")

# PRODUCTION CHECK: Restrict CORS origins instead of wildcards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Async Lock instantiation to guarantee mutual exclusion over shared global mutable model variables
state_mutation_lock = asyncio.Lock()

SECURE_WORKSPACE_ROOT = Path.home().resolve()

def validate_secure_path(user_input_path: str) -> Path:
    target_path = Path(user_input_path).resolve()
    # Use is_relative_to for a robust, cross-platform check (Python 3.9+)
    if not target_path.is_relative_to(SECURE_WORKSPACE_ROOT):
        raise HTTPException(status_code=403, detail="Access Denied: Path exits secure sandbox boundary.")
    return target_path

class ModelAttachRequest(BaseModel):
    path: str

@app.get("/health-and-capability-check")
async def health_check():
    return {
        "system": "Shiva AGI Gateway",
        "cognitive_state": runtime_manager.get_cognitive_state(),
        "registered_capabilities": capability_registry.list_capabilities()
    }

@app.post("/frankenMerging")
async def attach_model(request: ModelAttachRequest):
    secure_path = validate_secure_path(request.path)
    async with state_mutation_lock:
        result = attachment_manager.scan_model_directory(str(secure_path))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/list-of-models")
async def list_models():
    return {"models": attachment_manager.list_attached_models()}

@app.post("/models/{model_id}/merge")
async def trigger_frankenmerge(model_id: str):
    """Upgraded clean REST endpoint route convention replacing interpolated path binding."""
    async with state_mutation_lock:
        result = merge_manager.perform_merge(model_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@app.post("/execute")
async def execute_directive(request: ExecutionDirective):
    async with state_mutation_lock:
        try:
            return execution_manager.process_directive(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
