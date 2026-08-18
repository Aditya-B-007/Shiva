# Shiva 2.0 — Python Binding Module
#
# Python C-types binding module for Shiva 2.0 Sub-Millisecond Autonomous Control Engine.
# Provides Pythonic ctypes interface for loading `libshiva.so` / `libshiva.dylib` / `shiva.dll`.

import ctypes
import os
import sys
from typing import Optional, List, Tuple

# Define SystemInputDTO C-struct
class SystemInputDTO(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_float * 64),
        ("setpoint", ctypes.c_float * 32),
        ("state_stack", ctypes.c_float * 64),
        ("action_stack", ctypes.c_float * 32),
        ("hard_boundaries", ctypes.c_uint8 * 32),
        ("previous_rewards", ctypes.c_float),
        ("timestep", ctypes.c_uint64),
    ]

# Define ShivaOutputDTO C-struct
class ShivaOutputDTO(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_float * 64),
        ("reward", ctypes.c_float),
        ("mask", ctypes.c_uint8 * 32),
        ("final_action", ctypes.c_float * 32),
    ]

def _load_library() -> ctypes.CDLL:
    """Find and load the Shiva compiled C-dylib library."""
    lib_names = [
        "libshiva.dylib",  # macOS
        "libshiva.so",     # Linux
        "shiva.dll",       # Windows
    ]

    search_paths = [
        os.path.join(os.path.dirname(__file__), "../../target/release"),
        os.path.join(os.path.dirname(__file__), "../../target/debug"),
        os.path.dirname(__file__),
        "."
    ]

    for path in search_paths:
        for lib_name in lib_names:
            full_path = os.path.abspath(os.path.join(path, lib_name))
            if os.path.exists(full_path):
                return ctypes.CDLL(full_path)

    # Fallback to system search path
    for lib_name in lib_names:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue

    raise RuntimeError(
        "Could not find compiled Shiva dynamic library (libshiva.dylib / libshiva.so / shiva.dll).\n"
        "Please build the crate with `cargo build --release` first."
    )


class ShivaRuntime:
    """Pythonic wrapper around the Shiva 2.0 C-ABI runtime."""

    def __init__(self, matrix_rows: int = 20, min_signal: float = -1.0, max_signal: float = 1.0, lib_path: Optional[str] = None):
        if lib_path:
            self._lib = ctypes.CDLL(lib_path)
        else:
            self._lib = _load_library()

        # Configure C function prototypes
        self._lib.shiva_create.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_float]
        self._lib.shiva_create.restype = ctypes.c_void_p

        self._lib.shiva_destroy.argtypes = [ctypes.c_void_p]
        self._lib.shiva_destroy.restype = None

        self._lib.shiva_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(SystemInputDTO), ctypes.POINTER(ShivaOutputDTO)]
        self._lib.shiva_step.restype = ctypes.c_int32

        self._lib.shiva_default_input.argtypes = [ctypes.POINTER(SystemInputDTO)]
        self._lib.shiva_default_input.restype = None

        self._handle = self._lib.shiva_create(matrix_rows, min_signal, max_signal)
        if not self._handle:
            raise RuntimeError("Failed to create Shiva runtime engine instance.")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.shiva_destroy(self._handle)
            self._handle = None

    @staticmethod
    def create_default_input() -> SystemInputDTO:
        """Create a default SystemInputDTO initialized to safe values."""
        input_dto = SystemInputDTO()
        for i in range(64):
            input_dto.state[i] = 0.0
            input_dto.state_stack[i] = 0.0
        for i in range(32):
            input_dto.setpoint[i] = 0.0
            input_dto.action_stack[i] = 0.0
            input_dto.hard_boundaries[i] = 0
        input_dto.previous_rewards = 0.0
        input_dto.timestep = 0
        return input_dto

    def step(self, input_dto: SystemInputDTO) -> ShivaOutputDTO:
        """Execute a single 3-phase consensus cycle (< 1 ms)."""
        output_dto = ShivaOutputDTO()
        status = self._lib.shiva_step(self._handle, ctypes.byref(input_dto), ctypes.byref(output_dto))
        if status != 0:
            raise RuntimeError(f"Shiva runtime step failed with status code: {status}")
        return output_dto
