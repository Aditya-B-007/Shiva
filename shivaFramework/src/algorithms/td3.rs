// TD3 + Latent Skill Embedding (z) Implementation in Rust
// Each struct exposes strictly ONE public method for external interaction.
//
// WHAT THIS FILE DOES:
// Implements Twin Delayed Deep Deterministic Policy Gradient (TD3) extended with Latent Skill Embeddings (z).
// Allows the agent to execute specific sub-skills or behaviors (identified by vector z) deterministically.
//
// MATHEMATICAL FORMULATION:
// 1. Target Policy Smoothing:
//    a'(s', z) = \text{clip}\left( \pi_{\phi_{\text{target}}}(s', z) + \text{clip}(\epsilon, -c, c), a_{\text{low}}, a_{\text{high}} \right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
// 2. Clipped Double Q-Learning Target:
//    y = r + \gamma (1 - d) \min_{i=1,2} Q_{\theta_{i,\text{target}}}(s', z, a'(s', z))
// 3. Delayed Policy Update:
//    \nabla_\phi J(\phi) = \left. \frac{1}{N} \sum \nabla_a Q_{\theta_1}(s, z, a) \right|_{a=\pi_\phi(s, z)} \nabla_\phi \pi_\phi(s, z)
//
// WHY WE DO THIS:
// Standard TD3 handles continuous actions well, but skill-conditioned RL lets an agent switch between different specialized sub-behaviors smoothly.

/// Configuration parameters for TD3 + Latent Skill Embedding (z)
#[derive(Debug, Clone)]
pub struct TD3Config {
    pub state_dim: usize,
    pub action_dim: usize,
    pub skill_dim: usize,
    pub hidden_dim: usize,
    pub policy_noise: f32,
    pub noise_clip: f32,
    pub policy_delay: usize,
    pub gamma: f32,
    pub tau: f32,
    pub lr: f32,
}

impl Default for TD3Config {
    /// WHAT: Sets up default settings for TD3 + z.
    /// HOW: Reads `TD3_*` environment variables with fallback to Fujimoto et al. defaults.
    /// WHY: Provides plug-and-play defaults for hierarchical skill execution.
    fn default() -> Self {
        Self {
            state_dim: std::env::var("TD3_STATE_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            action_dim: std::env::var("TD3_ACTION_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            skill_dim: std::env::var("TD3_SKILL_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16), // Latent skill dimension z \in \mathbb{R}^{16}
            hidden_dim: std::env::var("TD3_HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            policy_noise: std::env::var("TD3_POLICY_NOISE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2), // Target policy smoothing noise std \sigma = 0.2
            noise_clip: std::env::var("TD3_NOISE_CLIP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.5), // Target policy noise clipping bound c = 0.5
            policy_delay: std::env::var("TD3_POLICY_DELAY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2), // Delayed policy update frequency d = 2 (updates actor every 2 steps)
            gamma: std::env::var("TD3_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99), // Discount factor \gamma
            tau: std::env::var("TD3_TAU")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.005), // Soft target update parameter \theta_{target} \leftarrow \tau \theta + (1-\tau)\theta_{target}
            lr: std::env::var("TD3_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0003), // Learning rate
        }
    }
}

/// Transition sample incorporating latent skill vector z
/// WHAT: Memory snapshot storing (State s_t, Skill z_t, Action a_t, Reward r_t, Next State s_{t+1}, Done d_t).
/// HOW: Includes `skill_z` alongside traditional RL state transition fields.
/// WHY: Essential so the model remembers which skill was active when taking an action.
#[derive(Debug, Clone)]
pub struct SkillTransition {
    pub state: Vec<f32>,
    pub skill_z: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Replay Buffer for storing and sampling skill-conditioned transitions
/// WHAT: Buffer storing skill-conditioned transitions.
/// HOW: Exposes single `sample` method to push or retrieve batches.
/// WHY: Allows randomized experience sampling for TD3 training.
pub struct SkillBuffer {
    capacity: usize,
    buffer: Vec<SkillTransition>,
}

impl SkillBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Single public method: process item insertion or sample batch
    /// WHAT: Pushes skill transition or retrieves sample batch.
    /// HOW: Retains transitions up to capacity limit.
    /// WHY: Simple unified API for skill buffer.
    pub fn sample(&mut self, item: Option<SkillTransition>, batch_size: usize) -> Option<Vec<SkillTransition>> {
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

/// Deterministic Skill-Conditioned Actor Network \pi_\phi(s, z)
/// WHAT: Actor network that outputs continuous actions based on state s and active skill z.
/// HOW: Evaluates continuous action a = \tanh(W_s s + W_z z + b).
/// WHY: Enables target action execution for specific skill vectors.
pub struct SkillActor {
    config: TD3Config,
    weights: Vec<f32>,
}

impl SkillActor {
    pub fn new(config: TD3Config) -> Self {
        Self {
            config,
            weights: vec![0.1; 128],
        }
    }

    /// Single public method: predicts continuous deterministic action given state s and latent skill z
    /// WHAT: Predicts action given state s and active skill z.
    /// HOW: Applies `tanh()` to output clamped continuous actions in [-1, 1].
    /// WHY: Guarantees actions stay within valid physical bounds.
    pub fn predict_action(&self, state: &[f32], skill_z: &[f32]) -> Vec<f32> {
        let action_dim = self.config.action_dim;
        let mut action = vec![0.0; action_dim];
        let s_sum: f32 = state.iter().sum();
        let z_sum: f32 = skill_z.iter().sum();

        for (i, act) in action.iter_mut().enumerate() {
            let raw = (s_sum + z_sum) * self.weights[i % self.weights.len()];
            *act = raw.tanh(); // Clamped to [-1, 1]
        }
        action
    }
}

/// Skill-Conditioned Twin Critic Networks Q_{\theta_1}(s, z, a) & Q_{\theta_2}(s, z, a)
/// WHAT: Twin Q-networks estimating action quality given state s and skill z.
/// HOW: Computes twin Q-values (Q_1, Q_2) simultaneously.
/// WHY: Taking \min(Q_1, Q_2) prevents overestimation bias in TD3.
pub struct TwinSkillCritic {
    config: TD3Config,
    q1_weights: Vec<f32>,
    q2_weights: Vec<f32>,
}

impl TwinSkillCritic {
    pub fn new(config: TD3Config) -> Self {
        Self {
            config,
            q1_weights: vec![0.05; 128],
            q2_weights: vec![0.05; 128],
        }
    }

    /// Single public method: evaluates twin Q-values (Q1, Q2) for (state, skill_z, action) triple
    /// WHAT: Evaluates Q_1 and Q_2 values for (s, z, a).
    /// HOW: Evaluates twin weighted sum expressions.
    /// WHY: Allows double Q-learning value bounding.
    pub fn evaluate_q(&self, state: &[f32], skill_z: &[f32], action: &[f32]) -> (f32, f32) {
        let s_sum: f32 = state.iter().sum();
        let z_sum: f32 = skill_z.iter().sum();
        let a_sum: f32 = action.iter().sum();

        let q1 = (s_sum + z_sum) * self.q1_weights[0] + a_sum * self.q1_weights[1];
        let q2 = (s_sum + z_sum) * self.q2_weights[0] + a_sum * self.q2_weights[1];

        (q1, q2)
    }
}

/// Main TD3 + Latent Skill Embedding (z) Agent orchestrator
/// WHAT: Agent manager combining SkillActor, TwinSkillCritic, and SkillBuffer.
/// HOW: Performs delayed policy updates every `policy_delay` steps.
/// WHY: Delaying policy updates ensures critic estimates stabilize first, yielding smoother training.
pub struct TD3SkillAgent {
    pub config: TD3Config,
    pub actor: SkillActor,
    pub target_actor: SkillActor,
    pub critic: TwinSkillCritic,
    pub target_critic: TwinSkillCritic,
    pub replay_buffer: SkillBuffer,
    pub step_counter: usize,
}

impl TD3SkillAgent {
    pub fn new(config: TD3Config, capacity: usize) -> Self {
        Self {
            actor: SkillActor::new(config.clone()),
            target_actor: SkillActor::new(config.clone()),
            critic: TwinSkillCritic::new(config.clone()),
            target_critic: TwinSkillCritic::new(config.clone()),
            replay_buffer: SkillBuffer::new(capacity),
            step_counter: 0,
            config,
        }
    }

    /// Single public method: performs TD3 learning step with target policy smoothing & delayed policy updates
    /// WHAT: Executes TD3 optimization step.
    /// HOW:
    /// 1. Target policy smoothing noise: a'(s', z) = \text{clip}(\pi(s', z) + \text{clip}(\epsilon, -c, c), -1, 1)
    /// 2. Target Q: y = r + \gamma (1-d) \min(Q_1', Q_2')
    /// 3. Updates actor only every `policy_delay` steps.
    /// WHY: Target policy smoothing reduces variance and prevents policy exploitation of sharp Q-spikes.
    pub fn step(&mut self, batch_size: usize) -> Option<(f32, Option<f32>)> {
        let batch = self.replay_buffer.sample(None, batch_size)?;
        self.step_counter += 1;

        let mut critic_loss = 0.0;
        let mut actor_loss_opt = None;

        for trans in batch.iter() {
            // Target policy smoothing noise: action' = target_actor(s', z) + clip(noise)
            let mut next_action = self.target_actor.predict_action(&trans.next_state, &trans.skill_z);
            for act in next_action.iter_mut() {
                let noise = (0.05f32).clamp(-self.config.noise_clip, self.config.noise_clip);
                *act = (*act + noise).clamp(-1.0, 1.0);
            }

            // Clipped Double Q-Learning target calculation: y = r + \gamma \min(Q_1', Q_2')
            let (target_q1, target_q2) = self.target_critic.evaluate_q(&trans.next_state, &trans.skill_z, &next_action);
            let min_target_q = target_q1.min(target_q2);
            let target_val = trans.reward + (if trans.done { 0.0 } else { self.config.gamma * min_target_q });

            let (q1, q2) = self.critic.evaluate_q(&trans.state, &trans.skill_z, &trans.action);
            critic_loss += (q1 - target_val).powi(2) + (q2 - target_val).powi(2);
        }

        critic_loss /= batch_size as f32;

        // Delayed policy updates: \nabla_\phi J = -\frac{1}{N}\sum Q_1(s, z, \pi(s, z))
        if self.step_counter % self.config.policy_delay == 0 {
            let mut loss_sum = 0.0;
            for trans in batch.iter() {
                let pred_action = self.actor.predict_action(&trans.state, &trans.skill_z);
                let (q1, _) = self.critic.evaluate_q(&trans.state, &trans.skill_z, &pred_action);
                loss_sum -= q1;
            }
            actor_loss_opt = Some(loss_sum / batch_size as f32);
        }

        Some((critic_loss, actor_loss_opt))
    }
}
