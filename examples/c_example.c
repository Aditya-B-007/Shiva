/*
 * Shiva 2.0 C Example — Autonomous Control Loop Integration
 *
 * Demonstrates how a C software stack (e.g. flight software, RTOS, microcontrollers)
 * integrates with Shiva 2.0 using the C-ABI.
 *
 * Compilation command:
 *   gcc -O3 c_example.c -I../bindings/c -L../target/release -lshiva -o c_example
 */

#include <stdio.h>
#include <stdlib.h>
#include "../bindings/c/shiva.h"

int main() {
    printf("=== Shiva 2.0 C API Control Loop ===\n");

    /* 1. Create Shiva runtime engine instance */
    ShivaHandle shiva = shiva_create(30, -1.0f, 1.0f);
    if (!shiva) {
        fprintf(stderr, "Failed to create Shiva runtime!\n");
        return 1;
    }
    printf("Shiva C runtime initialized successfully.\n");

    /* 2. Prepare sensor input packet */
    SystemInputDTO input;
    shiva_default_input(&input);

    /* 3. Run control loop for 5 timesteps */
    for (uint64_t step = 1; step <= 5; ++step) {
        input.timestep = step;
        for (int i = 0; i < 64; ++i) {
            input.state[i] = 0.1f * step;
        }
        input.previous_rewards = 1.0f;

        ShivaOutputDTO output;
        int32_t status = shiva_step(shiva, &input, &output);
        if (status != 0) {
            fprintf(stderr, "Error executing cycle step: %d\n", status);
            break;
        }

        printf("Timestep %02lu | Action[0]: %.4f | Reward: %.2f | Mask[0]: %u\n",
               (unsigned long)step, output.final_action[0], output.reward, output.mask[0]);
    }

    /* 4. Cleanup runtime memory */
    shiva_destroy(shiva);
    printf("=== Shiva C Loop Executed Cleanly ===\n");
    return 0;
}
