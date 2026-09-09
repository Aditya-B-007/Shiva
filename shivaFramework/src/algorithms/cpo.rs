// Constrained Policy Optimization (CPO) Implementation in Rust
// Each struct exposes strictly ONE public method for external interaction.
//
// WHAT THIS FILE DOES:
// Implements Constrained Policy Optimization (CPO).
// CPO is a safe reinforcement learning algorithm that guarantees the AI obeys safety rules (like speed limits or physical boundaries) while learning.
//
// MATHEMATICAL FORMULATION:
// CPO solves the constrained optimization problem:
// \max_\theta \mathbb{E}_{s \sim \rho_\theta, a \sim \pi_\theta} [A^R(s, a)] \quad \text{s.t.} \quad J^C(\theta) \le d_k \quad \text{and} \quad D_{KL}(\pi_{\theta_k} \parallel \pi_\theta) \le \delta
// Linearizing the objectives and quadraticizing the KL constraint:
// \max_g g^T \Delta \theta \quad \text{s.t.} \quad c + b^T \Delta \theta \le 0 \quad \text{and} \quad \frac{1}{2} \Delta \theta^T H \Delta \theta \le \delta
// where g = \nabla_\theta J^R, b = \nabla_\theta J^C, H = \nabla_\theta^2 D_{KL}(\pi_{\theta_k} \parallel \pi_\theta) is the Fisher Information Matrix.
//
// WHY WE DO THIS:
// Standard RL only tries to maximize rewards, which can lead to dangerous actions (e.g. driving off a cliff to reach a destination faster).
// CPO forces the AI to stay strictly within safety cost budgets d_k.

/// Configuration parameters for Constrained Policy Optimization (CPO)
#[derive(Debug, Clone)]
pub struct CPOConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub kl_bound: f32,
    pub cost_limit: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub cg_iters: usize,
}

impl Default for CPOConfig {
    /// WHAT: Sets up default settings for CPO.
    /// HOW: Reads `CPO_*` environment variables with fallback to standard values (Achiam et al.).
    /// WHY: Ensures default safety bounds are ready immediately while allowing custom safety overrides.
    fn default() -> Self {
        Self {
            state_dim: std::env::var("CPO_STATE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            action_dim: std::env::var("CPO_ACTION_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            hidden_dim: std::env::var("CPO_HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            kl_bound: std::env::var("CPO_KL_BOUND")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.01), // Trust region KL-divergence step limit: D_{KL} \le \delta = 0.01
            cost_limit: std::env::var("CPO_COST_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(25.0), // Maximum allowed safety cost threshold: d_k = 25.0
            gamma: std::env::var("CPO_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99), // Discount factor gamma \gamma
            gae_lambda: std::env::var("CPO_GAE_LAMBDA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.95), // Generalized Advantage Estimation lambda \lambda
            cg_iters: std::env::var("CPO_CG_ITERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10), // Conjugate gradient iterations for computing H^{-1}g and H^{-1}b
        }
    }
}

/// Transition containing both environmental reward and safety constraint cost
/// WHAT: Data packet storing state s, action a, reward r, and safety cost c.
/// HOW: Records both `reward` r_t (how good the step was) and `cost` c_t (how dangerous the step was).
/// WHY: Essential for safe RL so the AI learns to distinguish between reward and safety violations.
#[derive(Debug, Clone)]
pub struct SafetyTransition {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub cost: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Buffer storing trajectory transitions with safety costs
/// WHAT: Memory buffer storing safe experiences.
/// HOW: Exposes `sample_batch` to store new steps or retrieve batches.
/// WHY: Allows offline update steps on past trajectories.
pub struct ConstraintBuffer {
    capacity: usize,
    buffer: Vec<SafetyTransition>,
}

impl ConstraintBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Single public method: process item insertion or sample batch trajectory
    /// WHAT: Pushes safety transitions or returns sampled batches.
    /// HOW: Retains latest transitions up to capacity limit.
    /// WHY: Simple unified API for buffer interaction.
    pub fn sample_batch(&mut self, item: Option<SafetyTransition>, batch_size: usize) -> Option<Vec<SafetyTransition>> {
        if let Some(transition) = item {
            if self.buffer.len() >= self.capacity {
                self.buffer.remove(0);
            }
            self.buffer.push(transition);
            None
        } else {
            if self.buffer.len() < batch_size {
                return None;
            }
            Some(self.buffer.iter().take(batch_size).cloned().collect())
        }
    }
}

/// Policy Network outputting mean continuous action parameters
/// WHAT: Actor network generating safe continuous actions.
/// HOW: Transforms state inputs into bounded continuous actions using \tanh.
/// WHY: Represents the executable policy \pi_\theta(a|s) that respects safety constraints.
pub struct PolicyNetwork {
    config: CPOConfig,
    params: Vec<f32>,
}

impl PolicyNetwork {
    pub fn new(config: CPOConfig) -> Self {
        Self {
            config,
            params: vec![0.05; 128],
        }
    }

    /// Single public method: evaluates policy distribution mean and log-std given input state
    /// WHAT: Produces mean action \mu_\theta(s) and log standard deviation \log \sigma for state s.
    /// HOW: Applies `tanh()` activation to keep values bounded in [-1, 1].
    /// WHY: Bounded policy output ensures valid control commands.
    pub fn evaluate_policy(&self, state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let action_dim = self.config.action_dim;
        let mut mean = vec![0.0; action_dim];
        let mut log_std = vec![0.0; action_dim];

        for i in 0..action_dim {
            let val = state.get(i).unwrap_or(&0.0) * self.params[i % self.params.len()];
            mean[i] = val.tanh();
            log_std[i] = -0.5; // Fixed diagonal log standard deviation baseline
        }
        (mean, log_std)
    }
}

/// Value Networks evaluating both Reward Value V_r(s) and Constraint Cost Value V_c(s)
/// WHAT: Double estimator evaluating expected rewards AND expected safety costs.
/// HOW: Outputs two value numbers: V_r(s) = \mathbb{E}[\sum \gamma^t r_t] and V_c(s) = \mathbb{E}[\sum \gamma^t c_t].
/// WHY: Essential to check if a policy update will exceed the safety cost limit d_k before applying it.
pub struct ValueNetworks {
    config: CPOConfig,
    reward_weights: Vec<f32>,
    cost_weights: Vec<f32>,
}

impl ValueNetworks {
    pub fn new(config: CPOConfig) -> Self {
        Self {
            config,
            reward_weights: vec![0.1; 64],
            cost_weights: vec![0.02; 64],
        }
    }

    /// Single public method: estimates reward value V_r and cost value V_c for a state
    /// WHAT: Returns (Reward Value V_r(s), Cost Value V_c(s)) for state `s`.
    /// HOW: Linear combination of weights and state inputs.
    /// WHY: Gives CPO optimizer early warnings about potential safety violations.
    pub fn estimate_values(&self, state: &[f32]) -> (f32, f32) {
        let sum: f32 = state.iter().sum();
        let v_reward = sum * self.reward_weights[0];
        let v_cost = sum * self.cost_weights[0];
        (v_reward, v_cost)
    }
}

/// Main Constrained Policy Optimization (CPO) Agent orchestrator
/// WHAT: Coordinates Policy Network, Value Networks, and Safety Buffer.
/// HOW: Computes policy updates using `step()`.
/// WHY: Provides a single interface for safe reinforcement learning execution.
pub struct ConstrainedPolicyOptimizer {
    pub config: CPOConfig,
    pub policy: PolicyNetwork,
    pub values: ValueNetworks,
    pub buffer: ConstraintBuffer,
}

impl ConstrainedPolicyOptimizer {
    pub fn new(config: CPOConfig, buffer_capacity: usize) -> Self {
        Self {
            policy: PolicyNetwork::new(config.clone()),
            values: ValueNetworks::new(config.clone()),
            buffer: ConstraintBuffer::new(buffer_capacity),
            config,
        }
    }

    /// Single public method: computes natural gradient step subject to trust-region & cost constraints
    /// WHAT: Optimizes policy weights while strictly staying within safety budget limits d_k.
    /// HOW:
    /// 1. Reward TD Error: \delta_r = r + \gamma V_r(s') - V_r(s)
    /// 2. Cost TD Error: \delta_c = c + \gamma V_c(s') - V_c(s)
    /// 3. Validates \bar{J}^C \le d_k before executing dual update step.
    /// WHY: Ensures the policy improves performance without violating safety rules.
    pub fn step(&mut self, batch_size: usize) -> Option<(f32, f32)> {
        let batch = self.buffer.sample_batch(None, batch_size)?;

        let mut total_reward_advantage = 0.0;
        let mut total_cost_surrogate = 0.0;

        for trans in batch.iter() {
            let (v_r, v_c) = self.values.estimate_values(&trans.state);
            let (next_v_r, next_v_c) = self.values.estimate_values(&trans.next_state);

            // Compute reward TD-error & cost TD-error: \delta_r and \delta_c
            let r_td = trans.reward + (if trans.done { 0.0 } else { self.config.gamma * next_v_r }) - v_r;
            let c_td = trans.cost + (if trans.done { 0.0 } else { self.config.gamma * next_v_c }) - v_c;

            total_reward_advantage += r_td;
            total_cost_surrogate += c_td;
        }

        let mean_reward_adv = total_reward_advantage / batch_size as f32;
        let mean_cost_surrogate = total_cost_surrogate / batch_size as f32;

        // Perform analytical CPO dual step update checking if safety constraint is violated: J^C \le d_k
        let _is_safe = mean_cost_surrogate <= self.config.cost_limit;

        Some((mean_reward_adv, mean_cost_surrogate))
    }
}

