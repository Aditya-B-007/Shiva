# Shiva Desktop Packaging

This directory contains configuration scripts to package Shiva into a single-file executable (`.exe` on Windows, `.app` on macOS, binary on Linux) with an embedded Python environment.

## Requirements
Ensure you have installed the required base libraries before running the builder:
```bash
pip install -r ../requirements.txt
```

## How to Build
Run the build script from the root of the project:
```bash
python3 src/packaging/desktop/build.py
```
This compiles everything and puts the standalone executable inside the `src/packaging/desktop/dist/` directory.
Users do not need Python installed to double-click and run this compiled output.
