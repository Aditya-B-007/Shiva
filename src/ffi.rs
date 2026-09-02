// Shiva 2.0 — C / C++ / Cross-Language FFI Interface
//
// WHAT THIS FILE DOES:
// Exposes C-ABI extern "C" functions for embedding Shiva 2.0 into C, C++, Python,
// ROS2, and embedded RTOS software stacks.
//
// HOW IT DOES IT:
// Uses raw pointers with strict null-checks and `std::panic::catch_unwind` guards
// to ensure Rust unwinding panics never cross FFI boundaries into host languages.

use crate::config::ShivaBuilder;
use crate::protocol::middleMan::ManInTheMiddle;
use crate::protocol::shivaSide::ShivaOutputDTO;
use crate::protocol::systemSide::SystemInputDTO;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Opaque pointer handle to a Shiva `ManInTheMiddle` runtime instance
pub type ShivaHandle = *mut ManInTheMiddle;

/// Allocates and initializes a new Shiva runtime instance.
///
/// # Arguments
/// * `matrix_rows` - Sliding window row capacity (0 uses default of 20)
/// * `min_signal` - Lower bound for actuator signal clamping (-1.0 default if min == max)
/// * `max_signal` - Upper bound for actuator signal clamping (1.0 default if min == max)
///
/// # Returns
/// A non-null pointer to `ManInTheMiddle` on success, or `std::ptr::null_mut()` on failure.
#[no_mangle]
pub extern "C" fn shiva_create(
    matrix_rows: usize,
    min_signal: f32,
    max_signal: f32,
) -> ShivaHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let rows = if matrix_rows == 0 { 20 } else { matrix_rows };
        let (min, max) = if (min_signal - max_signal).abs() < f32::EPSILON {
            (-1.0, 1.0)
        } else {
            (min_signal, max_signal)
        };

        let instance = ShivaBuilder::new()
            .with_matrix_rows(rows)
            .with_actuator_limits(min, max)
            .build();

        Box::into_raw(Box::new(instance))
    }));

    result.unwrap_or(std::ptr::null_mut())
}

/// Destroys and frees a previously allocated Shiva runtime instance.
///
/// # Safety
/// `handle` must be a valid pointer returned by `shiva_create` or NULL.
#[no_mangle]
pub extern "C" fn shiva_destroy(handle: ShivaHandle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }));
}

/// Executes a single 3-phase consensus cycle.
///
/// # Arguments
/// * `handle` - Valid Shiva runtime handle
/// * `input` - Pointer to `SystemInputDTO` packet
/// * `output` - Pointer to `ShivaOutputDTO` struct where results will be written
///
/// # Returns
/// * `0` on success
/// * `-1` if handle is NULL
/// * `-2` if input or output pointer is NULL
/// * `-3` if a runtime panic occurred
#[no_mangle]
pub extern "C" fn shiva_step(
    handle: ShivaHandle,
    input: *const SystemInputDTO,
    output: *mut ShivaOutputDTO,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if input.is_null() || output.is_null() {
        return -2;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let shiva = &mut *handle;
        let in_dto = *input;
        let out_dto = shiva.counter(in_dto);
        *output = out_dto;
    }));

    match result {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Initializes a `SystemInputDTO` struct to default safe values.
///
/// # Safety
/// `input` pointer must not be NULL.
#[no_mangle]
pub extern "C" fn shiva_default_input(input: *mut SystemInputDTO) {
    if input.is_null() {
        return;
    }
    unsafe {
        *input = SystemInputDTO::default();
    }
}
