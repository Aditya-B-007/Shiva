# File Breakdown: Input, Perception Hooks, Output & Packaging Subsystems

This document details the entry point CLI scripts, workspace perception hooks, text-to-voice output synthesis, and desktop executable packaging configurations.

---

## 1. Main CLI Interface ([`src/input/cli.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/cli.py))

- **Lines 1–16**: Sets environment variables (`HF_HUB_DISABLE_PROGRESS_BARS`, `HF_HUB_DISABLE_SYMLINKS_WARNING`) to suppress verbose download logs. Configures base logging at `WARNING` level.
- **Lines 26–47 (`FallbackEmotionHandler`)**: Adapter pattern used when ML-backed emotion appraisal networks cannot be loaded. Proxies perception events directly into the memory engine using homeostatic state.
- **Lines 49–70 (`FallbackDecoder`)**: Deterministic decoder fallback guaranteeing CLI execution when PyTorch or local transformer model weights are absent. Synthesizes a structured `ThoughtDTO` explaining model unavailability.
- **Lines 72–139 (`initialize_cognitive_core`)**:
  - Instantiates `MemoryEngine` and `Homeostasis`.
  - Attempts to load `AppraisalEngine`, `FTTransformerFeatureEmbedding`, `CognitiveStateEncoder`, `EmotionDynamicsEngine`, and `EmotionalOrchestrator`. Catches exceptions and falls back to `FallbackEmotionHandler`.
  - Loads `TransformerDecoder` (`SmolLM2-360M-Instruct`) or falls back to `FallbackDecoder`.
  - Assembles `ReasoningScheduler` and returns the executive `Mothership` swarm coordinator.
- **Lines 141–171 (`handle_voice_input`)**: Launches a WebSocket server on `ws://localhost:8765` using `asyncio` and `websockets` to receive real-time audio transcription strings and push them onto `prompt_queue`.
- **Lines 181–204 (`run_query`)**: Executes a single reasoning problem on `Mothership`, prints formatted decision outputs and confidence scores, and optionally triggers Kokoro voice synthesis (`VoiceGenerator`).
- **Lines 205–278 (`main_async` & `main`)**: Parses arguments (`-q/--query`, `-d/--dir`, `-v/--voice-out`, `-i/--interactive`, `--voice-in`). Controls REPL prompt loop, background dream state toggling (`mothership.enter_dream_state()`), and WebSocket voice streaming.

---

## 2. Workspace Sensing Context ([`src/input/hook/workspace.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/hook/workspace.py))

- **Lines 7–15 (`__init__`)**: Takes a target workspace root directory string, resolves it into an absolute `Path`, and verifies existence.
- **Lines 17–30 (`_resolve_path`)**: Enforces strict directory sandboxing. Ensures target relative or absolute paths lie within `self.root_path`; raises `PermissionError` if an escape attempt outside the workspace root occurs.
- **Lines 32–47 (`list_dir`)**: Scans workspace contents using `os.scandir`, returning metadata lists (filename, `is_dir`, byte size, relative path).
- **Lines 49–67 (`view_file`)**: Reads UTF-8 file contents sliced between `start_line` and `end_line` (1-indexed inclusive).
- **Lines 69–75 (`write_to_file`)**: Writes content strings to target file, automatically creating missing parent directories.
- **Lines 76–90 (`replace_file_content`)**: Performs target text search and replacement; raises `ValueError` if target text is missing.
- **Lines 92–115 (`grep_search`)**: Scans workspace files using regex matching while ignoring VCS/build directories (`.git`, `__pycache__`, `build`, `dist`). Caps output at 100 search hits.
- **Lines 117–146 (`run_command`)**: Runs terminal commands inside `self.root_path` using `subprocess.run` with stdout/stderr capture and a 60-second execution timeout.

---

## 3. Perception Classification & Prompt Formatting ([`src/input/hook/perception.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/hook/perception.py))

- **Lines 9–30 (`PerceptionObservationFactory`)**: Classifies payloads (`ObservationKind.TEXT`, `BINARY_BYTES`, `STRUCTURED`, `UNKNOWN`), calculates payload byte sizes, extracts metadata keys, and builds compact observation summaries.
- **Lines 78–96 (`PerceptionPromptFormatter`)**: Converts `PerceptionBundleDTO` objects into standardized text blocks (`Query: ...`, `Observations: - device: kind=...; summary=...`) for injection into decoder reasoning prompts.

---

## 4. Voice Synthesis Engine ([`src/output/voice.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/output/voice.py))

- **Lines 4–14**: Conditionally imports Kokoro TTS (`from kokoro import KPipeline`), `soundfile`, and `numpy`. Sets `KOKORO_AVAILABLE` flag.
- **Lines 15–27 (`VoiceGenerator.__init__`)**: Configures voice identity (`af_heart`), language code (`a`), playback speed (`1.0`), and sample rate (`24000` Hz).
- **Lines 29–55 (`synthesize`)**: Lazy-loads `KPipeline`. Synthesizes audio segments from input text, concatenates audio arrays via NumPy, and writes output WAV files to disk via `soundfile.write`. Returns boolean status.

---

## 5. Packaging & Desktop Executable Scripts

### Desktop Packaging Script ([`src/packaging/desktop/build.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/packaging/desktop/build.py))
- **Lines 6–65 (`build_app`)**: Automates PyInstaller desktop executable creation. Automatically installs PyInstaller if absent, locates `src/input/cli.py`, cleans existing `dist/` and `build/` directories, and executes PyInstaller with `--onefile`, bundling `src/` modules.

### PyInstaller Spec File ([`Shiva.spec`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/Shiva.spec))
- Configures PyInstaller analysis, bundling `src/` source files, setting entry script to `src/input/cli.py`, and compiling a standalone desktop binary named `Shiva`.

### Package Setup ([`setup.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/setup.py))
- Configures setuptools packaging for Shiva (`name="shiva"`, `version="0.1.0"`, `packages=find_packages()`). Entry points include `shiva=src.input.cli:main`.
