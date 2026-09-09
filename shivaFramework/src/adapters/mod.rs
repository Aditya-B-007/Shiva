// Shiva 2.0 — Concrete Adapter Implementations
//
// WHAT THIS MODULE DOES:
// Provides default implementations of the InputAdapter and OutputAdapter
// traits defined in src/framework/adapter.rs.
//
// HOW IT IS ORGANIZED:
// ├── actuator_adapter  — Wraps ActuatorSignal behind OutputAdapter trait
// └── default_input     — Default InputAdapter with NaN/Inf cleaning

pub mod actuator_adapter;
pub mod default_input;

pub use actuator_adapter::ActuatorSignalAdapter;
pub use default_input::DefaultInputAdapter;
