// Shiva 2.0 — Environment Matrix Module
//
// WHAT THIS FILE DOES:
// Defines `EnvironmentMatrix`, a sliding-window memory queue storing 3-column rows
// `(action: [f32; 32], reward: f32, mask: [u8; 32])` with role-based access control.
//
// HOW IT DOES IT:
// - Configures max rows `x` from the `SHIVA_MATRIX_ROWS` environment variable (default: 20).
// - Enforces row capacity via `_popFromMatrix` when `pushRowToMatrix` is called by `shivaSide.rs`.
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
    /// Internal row buffer storing up to `max_rows`
    rows: Vec<MatrixRow>,
}

impl EnvironmentMatrix {
    /// 1a. __init__ / new: Matrix configuration.
    /// Number of columns = 3, max_rows = x from env var SHIVA_MATRIX_ROWS (default: 20).
    pub fn new() -> Self {
        let max_rows = env::var("SHIVA_MATRIX_ROWS")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(20);

        Self {
            num_columns: 3,
            max_rows,
            rows: Vec::with_capacity(max_rows),
        }
    }

    /// Alias constructor matching requested `__init__` naming convention.
    pub fn __init__() -> Self {
        Self::new()
    }

    /// 1b. pushRowToMatrix (Public): Pushes a new row into the matrix.
    /// Pops the oldest row if capacity exceeds x.
    /// DESIGNATED FOR USE BY `src/protocol/shivaSide.rs` ONLY.
    pub fn pushRowToMatrix(&mut self, action: [f32; 32], reward: f32, mask: [u8; 32]) {
        let new_row = MatrixRow {
            action,
            reward,
            mask,
        };

        self.rows.push(new_row);

        // If matrix length exceeds max_rows x, pop the oldest row
        if self.rows.len() > self.max_rows {
            self._popFromMatrix();
        }
    }

    /// Idiomatic snake_case alias for `pushRowToMatrix`.
    pub fn push_row_to_matrix(&mut self, action: [f32; 32], reward: f32, mask: [u8; 32]) {
        self.pushRowToMatrix(action, reward, mask);
    }

    /// 1c. _popFromMatrix (Private): Removes the oldest (x+1) row from the matrix.
    /// Called internally by `pushRowToMatrix`.
    fn _popFromMatrix(&mut self) {
        if !self.rows.is_empty() {
            self.rows.remove(0);
        }
    }

    /// Idiomatic snake_case helper calling `_popFromMatrix`.
    #[allow(dead_code)]
    fn pop_from_matrix(&mut self) {
        self._popFromMatrix();
    }

    /// 1d. _accessPolicy (Private): Defines access permissions for each NodeType.
    /// Returns the NodeAccessRule governing read access for the given node.
    fn _accessPolicy(&self, node_type: NodeType) -> NodeAccessRule {
        match node_type {
            // FailureEngine requires safety mask and latest reward to check anomaly triggers
            NodeType::FailureEngine => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: true,
                max_readable_rows: self.max_rows,
            },
            // FastDecision (SAC) reads action history and reward
            NodeType::FastDecision => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            // LongVision (IQN) reads long reward & action trajectory for tail risk evaluation
            NodeType::LongVision => NodeAccessRule {
                can_read_action: true,
                can_read_reward: true,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            // Explorer (TD3+z) reads action history for skill drift compensation
            NodeType::Explorer => NodeAccessRule {
                can_read_action: true,
                can_read_reward: false,
                can_read_mask: false,
                max_readable_rows: self.max_rows,
            },
            // GuardRail (CPO) reads hardware safety mask and recent action
            NodeType::GuardRail => NodeAccessRule {
                can_read_action: true,
                can_read_reward: false,
                can_read_mask: true,
                max_readable_rows: 5, // Limited window for slew rate checks
            },
            // Protocol ingestion node full system access
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
    pub fn readMatrix(&self, node_type: NodeType) -> Result<Vec<MatrixRow>, MatrixAccessError> {
        let policy = self._accessPolicy(node_type);

        // Fetch up to max_readable_rows from the end of the sliding window
        let start_idx = self.rows.len().saturating_sub(policy.max_readable_rows);
        let allowed_slice = &self.rows[start_idx..];

        // Filter columns according to policy permissions
        let filtered_rows: Vec<MatrixRow> = allowed_slice
            .iter()
            .map(|row| MatrixRow {
                action: if policy.can_read_action { row.action } else { [0.0; 32] },
                reward: if policy.can_read_reward { row.reward } else { 0.0 },
                mask: if policy.can_read_mask { row.mask } else { [0; 32] },
            })
            .collect();

        Ok(filtered_rows)
    }

    /// Idiomatic snake_case alias for `readMatrix`.
    pub fn read_matrix(&self, node_type: NodeType) -> Result<Vec<MatrixRow>, MatrixAccessError> {
        self.readMatrix(node_type)
    }

    /// Helper: returns the current number of rows stored in matrix.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Helper: checks if matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl Default for EnvironmentMatrix {
    fn default() -> Self {
        Self::new()
    }
}
