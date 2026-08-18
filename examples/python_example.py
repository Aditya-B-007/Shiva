#!/usr/bin/env python3
"""
Shiva 2.0 Python Example — Real-Time Control Loop

Demonstrates Python integration using the `shiva.py` C-types binding wrapper.
"""

import sys
import os
import time

# Ensure bindings/python is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../bindings/python"))

try:
    from shiva import ShivaRuntime, SystemInputDTO
except ImportError:
    print("Could not import shiva.py. Ensure bindings/python is accessible.")
    sys.exit(1)


def main():
    print("=== Shiva 2.0 Python Autonomous Control Loop ===")

    # 1. Initialize runtime
    # Note: Requires libshiva compiled via `cargo build`
    try:
        shiva = ShivaRuntime(matrix_rows=30, min_signal=-1.0, max_signal=1.0)
    except Exception as e:
        print(f"Initialization Note: {e}")
        print("Build libshiva with `cargo build --release` to run against the native binary.")
        return

    print("Shiva Python runtime initialized successfully.")

    # 2. Execute 5-step control loop
    input_dto = ShivaRuntime.create_default_input()

    for step in range(1, 6):
        input_dto.timestep = step
        for i in range(64):
            input_dto.state[i] = 0.1 * step
        input_dto.previous_rewards = 1.0

        output_dto = shiva.step(input_dto)

        print(
            f"Timestep {step:02d} | Action[0]: {output_dto.final_action[0]:.4f} | "
            f"Reward: {output_dto.reward:.2f} | Mask[0]: {output_dto.mask[0]}"
        )

    print("=== Shiva Python Loop Executed Cleanly ===")


if __name__ == "__main__":
    main()
