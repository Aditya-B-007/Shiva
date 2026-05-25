from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .capability import capability_registry
from .runtime import runtime_manager
from .attachment import attachment_manager
from .merge import merge_manager
from .execution import execution_manager, ExecutionDirective
import uvicorn

app = FastAPI(title="Shiva AGI Local Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    result = attachment_manager.scan_model_directory(request.path)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

@app.get("/list-of-models")
async def list_models():
    return {"models": attachment_manager.list_attached_models()}

@app.get("/{model_id}ModelMerge")
async def trigger_frankenmerge(model_id: str):
    result = merge_manager.perform_merge(model_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@app.post("/execute")
async def execute_directive(request: ExecutionDirective):
    try:
        return execution_manager.process_directive(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5123)
