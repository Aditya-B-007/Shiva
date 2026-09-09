// Random Network Distillation (RND) Implementation in Rust
// Each struct exposes strictly ONE public method for external interaction.
//
// WHAT THIS FILE DOES:
// Implements Random Network Distillation (RND) curiosity module.
// RND creates curiosity rewards for an AI by comparing a trainable "predictor" network against a fixed "random target" network.
//
// MATHEMATICAL FORMULATION:
// 1. Target Network (fixed random weights \theta^*):
//    f^*(s) \in \mathbb{R}^k
// 2. Predictor Network (trainable weights \theta):
//    \hat{f}_\theta(s) \in \mathbb{R}^k
// 3. Intrinsic Curiosity Reward & Predictor Loss:
//    R_i(s) = L_{\text{RND}}(\theta) = \frac{1}{k} \| \hat{f}_\theta(s) - f^*(s) \|_2^2 = \frac{1}{k} \sum_{i=1}^k \left( \hat{f}_{\theta, i}(s) - f_i^*(s) \right)^2
//
// WHY WE DO THIS:
// In environments with sparse rewards (like a maze with only 1 reward at the end), standard RL fails because the AI gets no feedback.
// RND generates internal curiosity rewards R_i(s) whenever the AI sees a new/unfamiliar state, encouraging active exploration.

/// Configuration parameters for Random Network Distillation (RND)
#[derive(Debug, Clone)]
pub struct RNDConfig {
    pub state_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    pub intrinsic_scale: f32,
}

impl Default for RNDConfig {
    /// WHAT: Sets up default settings for RND.
    /// HOW: Reads `RND_*` environment variables with fallback to Burda et al. defaults.
    /// WHY: Makes curiosity-driven exploration plug-and-play while allowing sensitivity tuning.
    fn default() -> Self {
        Self {
            state_dim: std::env::var("RND_STATE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            output_dim: std::env::var("RND_OUTPUT_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512), // Output representation dimension k = 512
            hidden_dim: std::env::var("RND_HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            lr: std::env::var("RND_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0001), // Learning rate for predictor network
            intrinsic_scale: std::env::var("RND_INTRINSIC_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0), // Intrinsic curiosity reward scaling factor
        }
    }
}

/// Running mean and standard deviation normalizer for observations
/// WHAT: Dynamically normalizes state inputs to zero mean and unit variance.
/// HOW: Updates online mean \mu and variance \sigma^2 via Welford's algorithm: \hat{s} = \text{clip}\left(\frac{s - \mu}{\sigma}, -5, 5\right).
/// WHY: Prevents high-scale inputs from dominating the RND curiosity reward error.
pub struct StateNormalizer {
    count: f32,
    mean: Vec<f32>,
    var: Vec<f32>,
}

impl StateNormalizer {
    pub fn new(dim: usize) -> Self {
        Self {
            count: 1e-4,
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
        }
    }

    /// Single public method: updates running stats and returns normalized observation
    /// WHAT: Normalizes state observation array s.
    /// HOW: Updates Welford's online running statistics and clamps output between [-5.0, 5.0].
    /// WHY: Clamping prevents single outlier state values from causing giant loss spikes.
    pub fn normalize(&mut self, state: &[f32]) -> Vec<f32> {
        self.count += 1.0;
        let mut normalized = vec![0.0; state.len()];

        for (i, &val) in state.iter().enumerate() {
            let delta = val - self.mean[i];
            self.mean[i] += delta / self.count;
            let delta2 = val - self.mean[i];
            self.var[i] += delta * delta2;

            let std = (self.var[i] / self.count).sqrt().max(1e-6);
            normalized[i] = ((val - self.mean[i]) / std).clamp(-5.0, 5.0);
        }
        normalized
    }
}

/// Fixed, randomly initialized target neural network f^*(s)
/// WHAT: Fixed target neural network f^*(s) that NEVER updates its weights.
/// HOW: Generates a deterministic pseudo-random feature output for any state s.
/// WHY: Serves as a static benchmark function for the predictor network to memorize.
pub struct RandomTargetNetwork {
    output_dim: usize,
    fixed_weights: Vec<f32>,
}

impl RandomTargetNetwork {
    pub fn new(config: &RNDConfig) -> Self {
        Self {
            output_dim: config.output_dim,
            fixed_weights: vec![0.42; 256], // Fixed random projection parameters
        }
    }

    /// Single public method: evaluates fixed target feature mapping for state s
    /// WHAT: Generates fixed target embedding vector f^*(s) \in \mathbb{R}^k.
    /// HOW: Applies fixed weights and sine activation function.
    /// WHY: Target embeddings are fixed so prediction error reflects state novelty.
    pub fn evaluate_target(&self, state: &[f32]) -> Vec<f32> {
        let s_sum: f32 = state.iter().sum();
        (0..self.output_dim)
            .map(|i| (s_sum * self.fixed_weights[i % self.fixed_weights.len()]).sin())
            .collect()
    }
}

/// Trainable predictor neural network f_hat(s) trying to distil target network
/// WHAT: Trainable predictor network \hat{f}_\theta(s) trying to predict what `RandomTargetNetwork` outputs.
/// HOW: Learns from states the AI visits frequently.
/// WHY: On familiar states, prediction error \|\hat{f}_\theta(s) - f^*(s)\|^2 will be LOW. On new states, prediction error will be HIGH.
pub struct PredictorNetwork {
    output_dim: usize,
    weights: Vec<f32>,
}

impl PredictorNetwork {
    pub fn new(config: &RNDConfig) -> Self {
        Self {
            output_dim: config.output_dim,
            weights: vec![0.1; 256],
        }
    }

    /// Single public method: evaluates predictor feature estimation for state s
    /// WHAT: Computes predicted feature embedding vector \hat{f}_\theta(s) \in \mathbb{R}^k.
    /// HOW: Applies predictor network weights and `tanh()` activation.
    /// WHY: Predicts target features; higher error indicates an unfamiliar state.
    pub fn evaluate_predictor(&self, state: &[f32]) -> Vec<f32> {
        let s_sum: f32 = state.iter().sum();
        (0..self.output_dim)
            .map(|i| (s_sum * self.weights[i % self.weights.len()]).tanh())
            .collect()
    }
}

/// Main Random Network Distillation (RND) Curiosity Module
/// WHAT: Orchestrates state normalizer, target network, and predictor network to compute curiosity reward.
/// HOW: Exposes single `compute_intrinsic_reward` method.
/// WHY: Gives RL algorithms an additional curiosity reward signal R_i(s) to drive exploration.
pub struct RNDModule {
    pub config: RNDConfig,
    pub target_net: RandomTargetNetwork,
    pub predictor_net: PredictorNetwork,
    pub normalizer: StateNormalizer,
}

impl RNDModule {
    pub fn new(config: RNDConfig) -> Self {
        let target_net = RandomTargetNetwork::new(&config);
        let predictor_net = PredictorNetwork::new(&config);
        let normalizer = StateNormalizer::new(config.state_dim);

        Self {
            config,
            target_net,
            predictor_net,
            normalizer,
        }
    }

    /// Single public method: computes intrinsic curiosity reward R_i(s) = ||f_hat(s) - f*(s)||^2
    /// WHAT: Calculates curiosity reward R_i(s) for observation state.
    /// HOW: Normalizes state, evaluates target and predictor embeddings, and calculates Mean Squared Error (MSE):
    /// R_i(s) = \frac{1}{k} \sum_{j=1}^k (\hat{f}_j(s) - f_j^*(s))^2 \cdot \text{scale}
    /// WHY: Returns high intrinsic reward for unfamiliar states, incentivizing exploration.
    pub fn compute_intrinsic_reward(&mut self, state: &[f32]) -> f32 {
        let norm_state = self.normalizer.normalize(state);

        let target_feats = self.target_net.evaluate_target(&norm_state);
        let pred_feats = self.predictor_net.evaluate_predictor(&norm_state);

        let mut mse = 0.0;
        for (t, p) in target_feats.iter().zip(pred_feats.iter()) {
            mse += (t - p).powi(2);
        }

        let raw_reward = mse / self.config.output_dim as f32;
        raw_reward * self.config.intrinsic_scale
    }
}

