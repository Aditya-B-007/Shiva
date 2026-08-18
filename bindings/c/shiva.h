/*
 * Shiva 2.0 — C / C++ Header
 *
 * C99 / C++ Header for the Shiva 2.0 Sub-Millisecond Autonomous Control Engine.
 * Provides C-ABI compatible data structures and functions for embedding Shiva
 * into C, C++, ROS / ROS2, and RTOS real-time hardware stacks.
 */

#ifndef SHIVA_H
#define SHIVA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SystemInputDTO — Input telemetry packet received from hardware sensors.
 * Memory layout aligned to 64 bytes for cache-line & SIMD performance.
 */
typedef struct {
    float state[64];            /* Sensor telemetry / state observation S_t */
    float setpoint[32];         /* Target control setpoint / desired trajectory */
    float state_stack[64];      /* Recent state history window */
    float action_stack[32];     /* Recent action history window */
    uint8_t hard_boundaries[32];/* Hardware safety interlocks (1 = fault, 0 = legal) */
    float previous_rewards;     /* Sensor reward feedback scalar */
    uint64_t timestep;          /* Monotonic cycle timestamp */
} SystemInputDTO;

/**
 * ShivaOutputDTO — Safe control packet dispatched back to physical actuators.
 * Memory layout aligned to 64 bytes.
 */
typedef struct {
    float state[64];            /* Goal-conditioned processed state */
    float reward;               /* Evaluated step reward scalar */
    uint8_t mask[32];           /* Active safety rule bitmask */
    float final_action[32];     /* Dispatched motor / thruster commands a*_t */
} ShivaOutputDTO;

/** Opaque handle to a Shiva runtime engine instance */
typedef void* ShivaHandle;

/**
 * Creates and initializes a new Shiva runtime engine instance.
 *
 * @param matrix_rows Max sliding window matrix capacity (pass 0 for default 20)
 * @param min_signal Minimum actuator signal limit (e.g. -1.0)
 * @param max_signal Maximum actuator signal limit (e.g. 1.0)
 * @return Non-null handle on success, NULL on failure
 */
ShivaHandle shiva_create(size_t matrix_rows, float min_signal, float max_signal);

/**
 * Destroys and frees a previously allocated Shiva runtime instance.
 *
 * @param handle Valid Shiva runtime handle
 */
void shiva_destroy(ShivaHandle handle);

/**
 * Executes a single 3-phase consensus cycle (< 1 ms).
 *
 * @param handle Valid Shiva runtime handle
 * @param input Pointer to SystemInputDTO sensor packet
 * @param output Pointer to ShivaOutputDTO packet where results are written
 * @return 0 on success, negative integer error code on failure
 */
int32_t shiva_step(ShivaHandle handle, const SystemInputDTO* input, ShivaOutputDTO* output);

/**
 * Initializes a SystemInputDTO struct with safe zero-default values.
 *
 * @param input Pointer to SystemInputDTO struct to initialize
 */
void shiva_default_input(SystemInputDTO* input);

#ifdef __cplusplus
}
#endif

#endif /* SHIVA_H */
