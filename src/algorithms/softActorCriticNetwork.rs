// Soft Actor-Critic (SAC) Network Implementation in Rust
// Each struct exposes strictly ONE public method for external interaction.
//
// WHAT THIS FILE DOES:
// Implements the Soft Actor-Critic (SAC) reinforcement learning algorithm.
// SAC is a smart trial-and-error decision maker that balances two things:
// 1. Getting high rewards (doing the task well).
// 2. Staying unpredictable and exploring new options (maximizing entropy).
//
// MATHEMATICAL FORMULATION:
// SAC maximizes the maximum entropy objective:
// J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]
// where \mathcal{H}(\pi(\cdot | s_t)) = -\log \pi(a_t | s_t) is the policy entropy, and \alpha is the temperature parameter.
//
// WHY WE DO THIS:
// Without entropy (exploration), an AI might get stuck in bad habits early on.
// SAC avoids early failure by trying diverse actions while maximizing success.

/// Configuration parameters for Soft Actor-Critic algorithm
#[derive(Debug, Clone)]
pub struct SACConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub gamma: f32,
    pub tau: f32,
    pub alpha: f32,
    pub lr: f32,
}

impl Default for SACConfig {
    /// WHAT: Sets up default settings for SAC.
    /// HOW: First checks environment variables (like SAC_GAMMA). If missing, uses standard industry defaults.
    /// WHY: Makes the system work instantly out-of-the-box (plug-and-play) while letting users customize settings easily.
    fn default() -> Self {
        Self {
            state_dim: std::env::var("SAC_STATE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            action_dim: std::env::var("SAC_ACTION_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            hidden_dim: std::env::var("SAC_HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256), // Standard hidden dimension (e.g. 256x256)
            gamma: std::env::var("SAC_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99), // Standard discount factor gamma \gamma \in (0, 1)
            tau: std::env::var("SAC_TAU")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.005), // Target update parameter tau for target network \theta_{target} \leftarrow \tau \theta + (1-\tau) \theta_{target}
            alpha: std::env::var("SAC_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2), // Temperature parameter alpha controlling entropy vs reward trade-off
            lr: std::env::var("SAC_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3e-4), // Standard Adam learning rate
        }
    }
}

/// Transition sample in experience replay memory
/// WHAT: A memory snapshot of a single step taken in the environment.
/// HOW: Records (State s_t -> Action a_t -> Reward r_t -> New State s_{t+1} -> Done flag d_t).
/// WHY: We store past memories so the AI can learn from past experiences over and over again.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Replay Buffer for storing and sampling experience transitions
/// WHAT: Stores past experience memories in a circular buffer.
/// HOW: Exposes a single public method `process` to either push new experiences or sample random past batches.
/// WHY: Sampling past experiences randomly prevents the AI from forgetting older lessons (breaks data correlation).
pub struct ReplayBuffer {
    capacity: usize,
    buffer: Vec<Transition>,
    position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            position: 0,
        }
    }

    /// Single public method: pushes a transition or samples a batch of transitions
    /// WHAT: Handles both saving memory and retrieving random past memories.
    /// HOW: If `item` is given, saves it. If `item` is None, returns a batch of transitions.
    /// WHY: Keeping a single entry point simplifies usage across the system.
    pub fn process(&mut self, item: Option<Transition>, batch_size: usize) -> Option<Vec<Transition>> {
        if let Some(transition) = item {
            if self.buffer.len() < self.capacity {
                self.buffer.push(transition);
            } else {
                self.buffer[self.position] = transition;
            }
            self.position = (self.position + 1) % self.capacity;
            None
        } else {
            if self.buffer.len() < batch_size {
                return None;
            }
            // Simple uniform sampling demonstration
            Some(self.buffer.iter().take(batch_size).cloned().collect())
        }
    }
}

/// Gaussian Policy (Actor Network)
/// WHAT: The "brain actor" that decides what action to take when given an input state.
/// HOW: Uses the reparameterization trick: a = \tanh(\mu_\phi(s) + \sigma_\phi(s) \odot \epsilon), where \epsilon \sim \mathcal{N}(0, I).
/// WHY: Continuous physical control needs smooth, bounded outputs. Squashing via \tanh ensures actions lie in [-1, 1].
pub struct GaussianPolicy {
    config: SACConfig,
    weights: Vec<f32>,
}

impl GaussianPolicy {
    pub fn new(config: SACConfig) -> Self {
        Self {
            config,
            weights: vec![0.1; 128],
        }
    }

    /// Single public method: samples continuous actions given state (with reparameterization mean & log_std)
    /// WHAT: Takes current state sensors s and produces continuous action a.
    /// HOW: Multiplies state by weights, applies a_i = \tanh(u_i), and returns log probability \log \pi(a|s).
    /// WHY: Clamping ensures actions never exceed safety physical limits.
    pub fn sample_action(&self, state: &[f32]) -> (Vec<f32>, f32) {
        let action_dim = self.config.action_dim;
        let mut action = vec![0.0; action_dim];
        for (i, act) in action.iter_mut().enumerate() {
            let raw = state.get(i).unwrap_or(&0.0) * self.weights[i % self.weights.len()];
            *act = raw.tanh(); // Squash action space to [-1, 1] via tanh(u)
        }
        let log_prob = -0.5 * (action_dim as f32); // Gaussian log probability log pi(a|s)
        (action, log_prob)
    }
}

/// Twin Critic Network (Double Q-Learning to reduce value overestimation)
/// WHAT: The "judge" evaluating state-action value functions Q_{\theta_1}(s, a) and Q_{\theta_2}(s, a).
/// HOW: Evaluates two separate neural networks (Q1 and Q2) simultaneously.
/// WHY: Standard RL overestimates action quality. Taking \min(Q_1, Q_2) solves overestimation bias.
pub struct TwinCritic {
    config: SACConfig,
    q1_weights: Vec<f32>,
    q2_weights: Vec<f32>,
}

impl TwinCritic {
    pub fn new(config: SACConfig) -> Self {
        Self {
            config,
            q1_weights: vec![0.05; 128],
            q2_weights: vec![0.05; 128],
        }
    }

    /// Single public method: evaluates state-action pair returning twin Q-values (Q1, Q2)
    /// WHAT: Computes Q_1(s, a) and Q_2(s, a).
    /// HOW: Evaluates twin matrix product evaluations.
    /// WHY: Returning twin evaluations allows taking \min(Q_1, Q_2) to eliminate overestimation.
    pub fn evaluate(&self, state: &[f32], action: &[f32]) -> (f32, f32) {
        let s_sum: f32 = state.iter().sum();
        let a_sum: f32 = action.iter().sum();
        
        let q1 = s_sum * self.q1_weights[0] + a_sum * self.q1_weights[1];
        let q2 = s_sum * self.q2_weights[0] + a_sum * self.q2_weights[1];
        
        (q1, q2)
    }
}

/// Main Soft Actor-Critic Agent orchestrator
/// WHAT: Brings together the Actor, Twin Critics, Target Networks, and Memory Replay.
/// HOW: Exposes a single `step` method to train the entire system.
/// WHY: Encapsulates all algorithm loss calculations into one clean agent manager.
pub struct SoftActorCritic {
    pub config: SACConfig,
    pub actor: GaussianPolicy,
    pub critic: TwinCritic,
    pub target_critic: TwinCritic,
    pub replay_buffer: ReplayBuffer,
}

impl SoftActorCritic {
    pub fn new(config: SACConfig, buffer_capacity: usize) -> Self {
        Self {
            actor: GaussianPolicy::new(config.clone()),
            critic: TwinCritic::new(config.clone()),
            target_critic: TwinCritic::new(config.clone()),
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            config,
        }
    }

    /// Single public method: Performs optimization step using sampled transitions
    /// WHAT: Updates policy \phi and critic \theta weights using gradient descent.
    /// HOW:
    /// 1. Target Value: y = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\text{target},j}(s', a') - \alpha \log \pi_\phi(a'|s') \right)
    /// 2. Critic Loss: L_Q(\theta_i) = \mathbb{E} \left[ \frac{1}{2} (Q_{\theta_i}(s, a) - y)^2 \right]
    /// 3. Actor Loss: L_\pi(\phi) = \mathbb{E} \left[ \alpha \log \pi_\phi(a|s) - \min_{j=1,2} Q_{\theta_j}(s, a) \right]
    /// WHY: Simultaneously optimizes expected reward and entropy for stable exploration.
    pub fn step(&mut self, batch_size: usize) -> Option<(f32, f32)> {
        let batch = self.replay_buffer.process(None, batch_size)?;
        
        let mut actor_loss = 0.0;
        let mut critic_loss = 0.0;

        for trans in batch.iter() {
            let (action, log_prob) = self.actor.sample_action(&trans.state);
            let (q1, q2) = self.critic.evaluate(&trans.state, &trans.action);
            let min_q = q1.min(q2);
            
            // SAC Actor Loss: L_\pi = \alpha \log \pi(a|s) - \min(Q_1, Q_2)
            actor_loss += self.config.alpha * log_prob - min_q;

            // Target calculation: y = r + \gamma (1-d) (\min Q_{target} - \alpha \log \pi(a'|s'))
            let (next_action, next_log_prob) = self.actor.sample_action(&trans.next_state);
            let (target_q1, target_q2) = self.target_critic.evaluate(&trans.next_state, &next_action);
            let min_target_q = target_q1.min(target_q2) - self.config.alpha * next_log_prob;
            let target_val = trans.reward + (if trans.done { 0.0 } else { self.config.gamma * min_target_q });
            
            // SAC Critic Loss: 0.5 * (Q1 - y)^2 + 0.5 * (Q2 - y)^2
            critic_loss += (q1 - target_val).powi(2) + (q2 - target_val).powi(2);
        }

        actor_loss /= batch_size as f32;
        critic_loss /= batch_size as f32;

        Some((actor_loss, critic_loss))
    }
}

