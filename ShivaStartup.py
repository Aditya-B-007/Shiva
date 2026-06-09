import os
import sys
import uvicorn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
import builtins
builtins.os = os
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌌 SHIVA AGI: UNIVERSAL GATED COGNITIVE ENGINE ENGINE IGNITION SEQUENCE")
    print("="*70)
    print("[*] Initializing core components...")
    print("[*] Instantiating Flyweight Swarm structures...")
    print("[*] Connecting Local Gateway API Endpoint to Frontend Workspace...")
    print("="*70 + "\n")
    uvicorn.run(
        "managers.main:app", 
        host="127.0.0.1", 
        port=5123, 
        reload=True,
        log_level="info"
    )
