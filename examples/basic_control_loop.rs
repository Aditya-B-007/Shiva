// Shiva 2.0 Framework Example — Basic Control Loop
//
// Demonstrates how an end-user developer integrates the Shiva 2.0 framework:
// 1. Configure runtime parameters using `ShivaBuilder`
// 2. Continuous control loop invoking `counter(input)`

use shiva::prelude::*;

fn main() {
    println!("=== Shiva 2.0 Framework Autonomous Control Loop ===");

    // 1. Initialize framework using ShivaBuilder
    let mut shiva = ShivaBuilder::new()
        .with_matrix_rows(30)              // Override matrix depth to 30 rows
        .with_actuator_limits(-1.0, 1.0)   // Clamp actuator motor signals to [-1.0, 1.0]
        .build();

    println!("Shiva framework initialized successfully!");
    println!("Matrix Rows Capacity: {}", shiva.matrix.max_rows);

    // 2. Simulate continuous control loop for 5 timesteps
    for step in 1..=5 {
        // Construct incoming system input packet
        let input = SystemInputDTO {
            state: [0.1 * step as f32; 64],
            setpoint: [0.0; 32],
            state_stack: [0.0; 64],
            action_stack: [0.0; 32],
            hard_boundaries: [0; 32],
            previous_rewards: 1.0,
            timestep: step as u64,
        };

        // Execution Gateway: Single counter method call!
        let output: ShivaOutputDTO = shiva.counter(input);

        println!(
            "Timestep {:02} | Action[0]: {:.4} | Reward: {:.2} | Emergency Veto: {}",
            step,
            output.final_action[0],
            output.reward,
            output.mask[0] != 0
        );
    }

    println!("=== Control Loop Executed Cleanly ===");
}
