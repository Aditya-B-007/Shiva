import os
import sys
import subprocess
import shutil

def build_app():
    """
    Automates compiling the Shiva package into a single-file executable using PyInstaller.
    """
    print("[Desktop Build] Starting packaging process...")
    
    # 1. Install PyInstaller if not present
    try:
        import PyInstaller
    except ImportError:
        print("[Desktop Build] Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 2. Establish entry point path and directories
    # Define root source directory and output paths
    packaging_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(packaging_dir))  # Shiva/src/
    project_root = os.path.dirname(src_dir)  # Shiva/
    
    # Create a mock entry point script dynamically under build folder
    entry_point = os.path.join(packaging_dir, "shiva_entry.py")
    with open(entry_point, "w") as f:
        f.write("""
import sys
from src.body.perception import get_current_os
print(f"Shiva active. Operating System: {get_current_os()}")
print("Perception devices ready.")
""")

    print(f"[Desktop Build] Generated temporary entry script: {entry_point}")

    # 3. Formulate PyInstaller command parameters
    # --onefile: Bundles everything into a single executable
    # --name: Executable name
    # --add-data: Bundles Shiva modules into the final package
    dist_dir = os.path.join(packaging_dir, "dist")
    build_dir = os.path.join(packaging_dir, "build")
    
    # Clean up previous builds if present
    for d in [dist_dir, build_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    cmd = [
        "pyinstaller",
        "--onefile",
        "--name=Shiva",
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        # Add src/ directory contents
        f"--add-data={src_dir}{os.path.pathsep}src",
        entry_point
    ]

    print(f"[Desktop Build] Running command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd, cwd=project_root)
        print(f"\n[Desktop Build] Packaging Successful! Executable created under: {dist_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Desktop Build] Error building executable: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary entry file
        if os.path.exists(entry_point):
            os.remove(entry_point)

if __name__ == "__main__":
    build_app()
