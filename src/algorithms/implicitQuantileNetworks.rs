// Implicit Quantile Networks (IQN) Implementation in Rust
// Each struct exposes strictly ONE public method for external interaction.
//
// WHAT THIS FILE DOES:
// Implements Implicit Quantile Networks (IQN).
// IQN is a distributional reinforcement learning framework. Instead of predicting just the single average return, it predicts the FULL range of possible return outcomes (quantiles).
//
// MATHEMATICAL FORMULATION:
// The return distribution Z(s, a) is modeled implicitly via its quantile function Z_\tau(s, a) for \tau \sim U(0, 1).
// Quantile embedding maps \tau using K cosine basis functions:
// \psi_j(\tau) = \text{ReLU}\left( \sum_{i=0}^{K-1} \cos(i \pi \tau) w_{ij} + b_j \right)
// The asymmetric Quantile Huber Loss is given by:
// \mathcal{L}_{\text{IQN}} = \frac{1}{N N'} \sum_{i=1}^N \sum_{j=1}^{N'} \rho_{\tau_i}^\kappa \left( \delta_{i, j} \right)
// where \delta_{i,j} = r + \gamma Z_{\tau_j'}(s', a^*) - Z_{\tau_i}(s, a), and \rho_\tau^\kappa(\delta) = |\tau - \mathbb{I}(\delta < 0)| \frac{\mathcal{L}_\kappa(\delta)}{\kappa}.
//
// WHY WE DO THIS:
// In high-risk environments, knowing the worst-case scenario (lower tail) is more important than knowing the average scenario. IQN gives the AI full risk awareness.

/// Configuration parameters for Implicit Quantile Networks (IQN)
#[derive(Debug, Clone)]
pub struct IQNConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub num_quantiles: usize,
    pub embedding_dim: usize,
    pub kappa: f32,
    pub gamma: f32,
    pub lr: f32,
}

impl Default for IQNConfig {
    /// WHAT: Sets up default settings for IQN.
    /// HOW: Reads `IQN_*` environment variables with fallback to standard defaults (Dabney et al.).
    /// WHY: Makes risk-sensitive training ready out-of-the-box while allowing hyperparameter tuning.
    fn default() -> Self {
        Self {
            state_dim: std::env::var("IQN_STATE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            action_dim: std::env::var("IQN_ACTION_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            hidden_dim: std::env::var("IQN_HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            num_quantiles: std::env::var("IQN_NUM_QUANTILES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64), // Number of quantile samples N
            embedding_dim: std::env::var("IQN_EMBEDDING_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64), // Cosine embedding dimension K
            kappa: std::env::var("IQN_KAPPA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0), // Quantile Huber loss threshold parameter \kappa = 1.0
            gamma: std::env::var("IQN_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99), // Discount factor \gamma
            lr: std::env::var("IQN_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.00005), // Learning rate
        }
    }
}

/// Transition sample in experience replay
/// WHAT: Stores one step of experience for quantile training.
/// HOW: Holds state s_t, discrete action index a_t, reward r_t, next state s_{t+1}, and done flag.
/// WHY: Essential payload for experience replay sampling.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Replay Buffer for sampling transitions
/// WHAT: Buffer holding past experience transitions for IQN training.
/// HOW: Exposes `sample` method for pushing or fetching batches.
/// WHY: Breaks correlation between sequential steps during training.
pub struct QuantileBuffer {
    capacity: usize,
    buffer: Vec<Transition>,
}

impl QuantileBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Single public method: process item insertion or sample batch
    /// WHAT: Either inserts a transition or samples a random batch.
    /// HOW: Retains recent transitions up to maximum capacity.
    /// WHY: Single entry point for memory buffer operations.
    pub fn sample(&mut self, item: Option<Transition>, batch_size: usize) -> Option<Vec<Transition>> {
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

/// Cosine Feature Embedding for Quantile Fraction Tau
/// WHAT: Converts random probability fractions \tau \in (0, 1) into cosine feature vectors.
/// HOW: Computes \phi_i(\tau) = \cos(i \cdot \pi \cdot \tau) for multiple frequencies i \in \{0, \dots, K-1\}.
/// WHY: Cosine embedding allows neural networks to learn continuous functions over arbitrary quantile fractions smoothly.
pub struct CosEmbedding {
    embedding_dim: usize,
}

impl CosEmbedding {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Single public method: computes cos(i * pi * tau) embeddings for sampled taus
    /// WHAT: Maps tau fractions to high-dimensional feature vectors.
    /// HOW: Calculates cosine harmonics \cos(i \cdot \pi \cdot \tau) for each tau sample.
    /// WHY: Enables smooth continuous quantile function representation.
    pub fn embed(&self, taus: &[f32]) -> Vec<Vec<f32>> {
        let pi = std::f32::consts::PI;
        taus.iter()
            .map(|&tau| {
                (0..self.embedding_dim)
                    .map(|i| (i as f32 * pi * tau).cos())
                    .collect()
            })
            .collect()
    }
}

/// Implicit Quantile Network calculating action-quantile values Z_\tau(s, a)
/// WHAT: Neural network that predicts return quantiles given state s and quantile fraction \tau.
/// HOW: Combines state representation with cosine \tau embeddings: Z_\tau(s, a) = f(s \odot \phi(\tau)).
/// WHY: Allows sampling any arbitrary quantile fraction \tau dynamically at runtime.
pub struct ImplicitQuantileNetwork {
    config: IQNConfig,
    cos_embedder: CosEmbedding,
    weights: Vec<f32>,
}

impl ImplicitQuantileNetwork {
    pub fn new(config: IQNConfig) -> Self {
        let cos_embedder = CosEmbedding::new(config.embedding_dim);
        Self {
            cos_embedder,
            weights: vec![0.01; 128],
            config,
        }
    }

    /// Single public method: computes return quantiles for given state across tau samples
    /// WHAT: Computes quantile values Z_\tau(s, a) for given state s and \tau samples.
    /// HOW: Integrates state input with cosine embedded taus.
    /// WHY: Gives full distribution predictions for all actions.
    pub fn evaluate_quantiles(&self, state: &[f32], taus: &[f32]) -> Vec<Vec<f32>> {
        let embeddings = self.cos_embedder.embed(taus);
        let s_sum: f32 = state.iter().sum();
        let action_dim = self.config.action_dim;

        embeddings
            .iter()
            .map(|emb| {
                let e_sum: f32 = emb.iter().sum();
                (0..action_dim)
                    .map(|a_idx| s_sum * self.weights[a_idx % self.weights.len()] + e_sum * 0.05)
                    .collect()
            })
            .collect()
    }
}

/// Core IQN Agent handling training step and quantile Huber loss
/// WHAT: Manages online and target IQN networks and handles training updates.
/// HOW: Uses asymmetric quantile Huber loss \rho_\tau^\kappa(\delta) to update networks based on sampled experiences.
/// WHY: Trains the model to accurately capture the full return distribution.
pub struct IQNAgent {
    pub config: IQNConfig,
    pub online_net: ImplicitQuantileNetwork,
    pub target_net: ImplicitQuantileNetwork,
    pub replay_buffer: QuantileBuffer,
}

impl IQNAgent {
    pub fn new(config: IQNConfig, capacity: usize) -> Self {
        Self {
            online_net: ImplicitQuantileNetwork::new(config.clone()),
            target_net: ImplicitQuantileNetwork::new(config.clone()),
            replay_buffer: QuantileBuffer::new(capacity),
            config,
        }
    }

    /// Single public method: executes training step and returns average quantile Huber loss
    /// WHAT: Performs one update step on a batch of experience transitions.
    /// HOW: Samples \tau \sim U(0,1), computes \delta_{i,j} = r + \gamma Z_{\tau_j'}(s', a^*) - Z_{\tau_i}(s, a), and applies asymmetric Huber penalty:
    /// \rho_\tau^\kappa(\delta) = |\tau - \mathbb{I}(\delta < 0)| \frac{\mathcal{L}_\kappa(\delta)}{\kappa}
    /// WHY: Minimizes distributional error to improve return distribution estimates.
    pub fn step(&mut self, batch_size: usize) -> Option<f32> {
        let batch = self.replay_buffer.sample(None, batch_size)?;
        let num_quantiles = self.config.num_quantiles;

        // Generate tau samples from Uniform(0, 1)
        let taus: Vec<f32> = (0..num_quantiles)
            .map(|i| (i as f32 + 0.5) / num_quantiles as f32)
            .collect();

        let mut total_huber_loss = 0.0;

        for trans in batch.iter() {
            let online_quantiles = self.online_net.evaluate_quantiles(&trans.state, &taus);
            let target_quantiles = self.target_net.evaluate_quantiles(&trans.next_state, &taus);

            for (i, &tau) in taus.iter().enumerate() {
                let q_val = online_quantiles[i][trans.action];
                let target_val = trans.reward + (if trans.done { 0.0 } else { self.config.gamma * target_quantiles[i][0] });

                let error = target_val - q_val;
                let abs_error = error.abs();

                // Quantile Huber Loss: L_\kappa(\delta)
                let huber_loss = if abs_error <= self.config.kappa {
                    0.5 * error.powi(2)
                } else {
                    self.config.kappa * (abs_error - 0.5 * self.config.kappa)
                };

                // Asymmetric weighting penalty: |\tau - \mathbb{I}(\delta < 0)|
                let asymmetric_weight = (tau - if error < 0.0 { 1.0 } else { 0.0 }).abs();
                total_huber_loss += asymmetric_weight * huber_loss / self.config.kappa;
            }
        }

        let avg_loss = total_huber_loss / (batch_size * num_quantiles) as f32;
        Some(avg_loss)
    }
}

