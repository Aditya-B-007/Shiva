// Shiva 2.0 — Environment Matrix Module
//
// WHAT THIS FILE DOES:
// Defines `EnvironmentMatrix`, a sliding-window memory queue storing 3-column rows
// `(action: [f32; 32], reward: f32, mask: [u8; 32])` with role-based access control.
//
// HOW IT DOES IT:
// - Configures max rows `x` from the `SHIVA_MATRIX_ROWS` environment variable (default: 20).
// - Uses a fixed-capacity ring buffer to store rows without allocations.
// - Implements `_accessPolicy` to restrict read permissions per `NodeType`.

use std::env;

/// Enum representing the 5 Mothership Ensemble Node Types + Protocol Ingestion Node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    FailureEngine,
    FastDecision,
    LongVision,
    Explorer,
    GuardRail,
    ShivaProtocol,
}

/// Represents a single row in the Environment Matrix (3 columns: action, reward, mask).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct MatrixRow {
    /// Column 0: Action vector [f32; 32]
    pub action: [f32; 32],
    /// Column 1: Scalar Reward f32
    pub reward: f32,
    /// Column 2: Hardware Safety Mask [u8; 32]
    pub mask: [u8; 32],
}

impl Default for MatrixRow {
    fn default() -> Self {
        Self {
            action: [0.0; 32],
            reward: 0.0,
            mask: [0; 32],
        }
    }
}

/// Access permission policy for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeAccessRule {
    pub can_read_action: bool,
    pub can_read_reward: bool,
    pub can_read_mask: bool,
    pub max_readable_rows: usize,
}

/// Error type returned when node access policy is violated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixAccessError {
    UnauthorizedAccess(String),
}

/// EnvironmentMatrix — Sliding window state store with access policy enforcement.
#[derive(Debug, Clone)]
pub struct EnvironmentMatrix {
    /// Number of matrix columns (fixed to 3: action, reward, mask)
    pub num_columns: usize,
    /// Maximum capacity of rows x (read from SHIVA_MATRIX_ROWS env var, default 20)
    pub max_rows: usize,
    /// Pre-allocated ring buffer (allocated once at construction, never resized)
    buffer: Vec<MatrixRow>,
    /// Write cursor index (next position to write to)
    head: usize,
    /// Number of valid rows currently stored (0..=max_rows)
    len: usize,
}

impl EnvironmentMatrix {
    /// 1a. __init__ / new: Matrix configuration.
    /// Number of columns = 3, max_rows = x from env var SHIVA_MATRIX_ROWS (default: 20).
    pub fn new() -> Self {
        let max_rows = env::var("SHIVA_MATRIX_ROWS")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(20);
        let max_rows = if max_rows == 0 { 1 } else { max_rows }; // Prevent division by zero

        Self {
            num_columns: 3,
            max_rows,
            buffer: vec![MatrixRow::default(); max_rows],
            head: 0,
            len: 0,
        }
    }

    /// Alias constructor matching requested `__init__` naming convention.
    pub fn __init__() -> Self {
        Self::new()
    }

    /// Constructs an EnvironmentMatrix using explicit `ShivaConfig` parameters.
    pub fn from_config(config: &crate::config::ShivaConfig) -> Self {
        let max_rows = if config.matrix_rows == 0 { 1 } else { config.matrix_rows };
        Self {
            num_columns: 3,
            max_rows,
            buffer: vec![MatrixRow::default(); max_rows],
            head: 0,
            len: 0,
        }
    }


    /// 1b. pushRowToMatrix (Public): Pushes a new row into the matrix.
    /// DESIGNATED FOR USE BY `src/protocol/shivaSide.rs` ONLY.
    pub fn pushRowToMatrix(&mut self, action: [f32; 32], reward: f32, mask: [u8; 32]) {
        let idx = self.head % self.max_rows;
        self.buffer[idx] = MatrixRow { action, reward, mask };
        self.head += 1;
        if self.len < self.max_rows {
            self.len += 1;
        }
    }

    /// Idiomatic snake_case alias for `pushRowToMatrix`.
    pub fn push_row_to_matrix(&mut self, action: [f32; 32], reward: f32, mask: [u8; 32]) {
        self.pushRowToMatrix(action, reward, mask);
    }

    /// 1d. _accessPolicy (Private): Defines access permissions for each NodeType.
    /// Returns the NodeAccessRule governing read access for the given node.
    fn _accessPolicy(&self, node_type: NodeType) -> NodeAccessRule {
        match node_type {
            NodeType::FailureEngine => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: true,
                max_readable_rows: self.max_rows,
            },
            NodeType::FastDecision => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            NodeType::LongVision => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            NodeType::Explorer => NodeAccessRule {
                can_read_action: true,
                can_read_reward: false,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            NodeType::GuardRail => NodeAccessRule {
                can_read_action: true,
                can_read_reward: false,
                can_read_mask: true,
                max_readable_rows: 5,
            },
            NodeType::ShivaProtocol => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: true,
                max_readable_rows: self.max_rows,
            },
        }
    }

    /// 1e. readMatrix (Public): Takes node_type input, uses `_accessPolicy` to verify access,
    /// and returns the permitted rows filtered according to node permissions.
    /// NOTE: Allocates a new Vec. For hot-path, use `read_matrix_into`.
    pub fn readMatrix(&self, node_type: NodeType) -> Result<Vec<MatrixRow>, MatrixAccessError> {
        let policy = self._accessPolicy(node_type);
        let num_to_read = std::cmp::min(self.len, policy.max_readable_rows);
        
        let mut filtered_rows = Vec::with_capacity(num_to_read);
        
        for i in 0..num_to_read {
            let logical_idx = self.len - num_to_read + i;
            let actual_idx = (self.head + self.max_rows - self.len + logical_idx) % self.max_rows;
            let row = &self.buffer[actual_idx];
            
            filtered_rows.push(MatrixRow {
                action: if policy.can_read_action { row.action } else { [0.0; 32] },
                reward: if policy.can_read_reward { row.reward } else { 0.0 },
                mask: if policy.can_read_mask { row.mask } else { [0; 32] },
            });
        }
        
        Ok(filtered_rows)
    }

    /// Idiomatic snake_case alias for `readMatrix`.
    pub fn read_matrix(&self, node_type: NodeType) -> Result<Vec<MatrixRow>, MatrixAccessError> {
        self.readMatrix(node_type)
    }

    /// Zero-allocation read alternative that writes directly into a provided slice.
    /// Returns the number of rows written.
    pub fn read_matrix_into(&self, node_type: NodeType, dest: &mut [MatrixRow]) -> Result<usize, MatrixAccessError> {
        let policy = self._accessPolicy(node_type);
        let num_to_read = std::cmp::min(self.len, std::cmp::min(policy.max_readable_rows, dest.len()));
        
        for i in 0..num_to_read {
            let logical_idx = self.len - num_to_read + i;
            let actual_idx = (self.head + self.max_rows - self.len + logical_idx) % self.max_rows;
            let row = &self.buffer[actual_idx];
            
            dest[i] = MatrixRow {
                action: if policy.can_read_action { row.action } else { [0.0; 32] },
                reward: if policy.can_read_reward { row.reward } else { 0.0 },
                mask: if policy.can_read_mask { row.mask } else { [0; 32] },
            };
        }
        
        Ok(num_to_read)
    }

    /// Helper: returns the current number of rows stored in matrix.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Helper: checks if matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for EnvironmentMatrix {
    fn default() -> Self {
        Self::new()
    }
}
