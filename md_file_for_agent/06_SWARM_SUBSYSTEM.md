# File Breakdown: Distributed Swarm Intelligence & Reinforcement Learning

This document provides an exhaustive breakdown of the executive Mothership prefrontal controller, physical inverted pendulum stability regulator, specialized Cortical Columns, and Soft Actor-Critic (SAC) reinforcement learning policy.

---

## 1. Executive Prefrontal Controller & Stability Regulator ([`src/swarm/mothership.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py))

- **Lines 29–56 (`CognitiveStabilityRegulator`)**:
  - Physics-based physical model tracking cognitive instability as an **inverted physical pendulum**:
    $$\ddot{\theta} = \frac{g \sin\theta + F_{\text{disturbance}} \cos\theta - F_{\text{control}}}{L}$$
  - `apply_cognitive_perturbations()` (Lines 37–49): Perturbs angular deviation $\theta$ driven by uncertainty ($1.5 \times$), stress ($1.0 \times$), and decision conflict ($2.0 \times$).
  - `apply_stabilizing_cortex_action()` (Lines 50–56): Applies stabilizing control effort computed from high-confidence column decisions to restore equilibrium ($\theta \rightarrow 0$).
- **Lines 57–131 (`Mothership.__init__` & Dream State)**:
  - Initializes memory engine, emotion handler, reasoning scheduler, perception factories, stability regulator, and neural encoders.
  - `enter_dream_state()` / `exit_dream_state()` (Lines 101–131): Spawns background thread running memory sleep cycles during idle user input periods. Automatically suspended on incoming queries.
- **Lines 132–150 (`arbitrate_columns`)**:
  - Prefrontal scheduling: Always deploys `AnalyticalColumn` (ID 1) and `CreativeColumn` (ID 2).
  - Instability Alerts: If $|\theta| > 0.15 \text{ rad}$, dynamically schedules `RiskColumn` (ID 3). If $|\theta| > 0.35 \text{ rad}$, schedules `VerificationColumn` (ID 4).
- **Lines 151–339 (`solve_problem`)**:
  - Master solving loop executing up to `max_cycles` reasoning cycles:
  1. Encodes goal vector via BERT-large `Encoder`.
  2. Encodes state text into continuous vector representations.
  3. Arbitrates and executes active cortical columns concurrently.
  4. Computes Temporal Difference (TD) Error / Reward Prediction Error (RPE):
     $$\text{RPE} = R_t + \gamma V(S_{t+1}) - V(S_t)$$
  5. Trains neuro-symbolic value network (`_value_network`) via Adam optimizer MSE loss.
  6. Updates inverted pendulum instability $\theta$ using RPE perturbations and applies stabilizing control effort.
  7. Updates shared pheromone confidence map (`pheromone_map`) and applies evaporation.
  8. Early stopping: Terminates if $|\theta| < 0.05$ and top decision confidence exceeds $0.85$.
  9. Executes Monte Carlo credit assignment (`memory.assign_credit_for_episode()`).
- **Lines 340–375 (`_capture_perception_bundle`)**: Scans target workspace via `WorkspaceContext` (`list_dir`, keyword `grep_search`) and attaches observations to `PerceptionBundleDTO`.

---

## 2. Specialized Cortical Columns ([`src/swarm/cells.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/cells.py))

- **Lines 12–56 (`CorticalColumn` Base Class)**:
  - Connects memory engine, emotion handler, scheduler, scratchpad, chain-of-thought, `nodeProcessingEngine`, and local `SwarmSAC` RL agent.
- **Lines 57–159 (`activate`)**:
  1. Formulates goal-conditioned state vector (`[state_emb, goal_emb]`).
  2. Samples continuous latent action projection vector from SAC Actor network (`sac.actor.sample()`).
  3. Passes latent action vector to `Decoder` to guide model generation.
  4. Swarm Critic Validation (Lines 96–131): If decision contains Python code, generates validation test script, executes it in the local sandbox, and adjusts step reward ($+0.5$ on pass, $-0.5$ on failure).
  5. Stores transition experience in Replay Buffer (`sac.add_experience()`) and executes SAC parameter update step (`sac.update_parameters()`).
- **Lines 178–216 (Specialized Column Classes)**:
  - `AnalyticalColumn`: Focuses on pure logic and math (`temperature=0.1`, `top_p=0.85`).
  - `CreativeColumn`: Focuses on analogies and novel connection paths (`temperature=0.85`, `top_p=0.95`).
  - `RiskColumn`: Focuses on safety checks and failure modes (`temperature=0.2`, `top_p=0.90`).
  - `VerificationColumn`: Audits decisions for contradictions (`temperature=0.05`, `top_p=0.80`).

---

## 3. Soft Actor-Critic RL Policy ([`src/swarm/SwarmSAC.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/SwarmSAC.py))

- **Lines 9–42 (`SACActor`)**:
  - Policy network mapping 4096-dimensional state-goal vectors to continuous 2048-dimensional action projection vectors.
  - Outputs mean and log std with numerical clamping (`torch.clamp(log_std, -20, 2)`). Uses reparameterization trick (`rsample()`) and applies $\tanh$ log-probability correction.
- **Lines 44–67 (`SACCritic`)**:
  - Twin Q-networks ($Q_1, Q_2$) evaluating state-action pairs $(s, a)$ to mitigate positive bias in value estimation.
- **Lines 69–150 (`SwarmSAC`)**:
  - Continuous Soft Actor-Critic agent ($\gamma=0.99$, $\tau=0.005$, temperature $\alpha=0.2$).
  - Replay Buffer: Stores $(s, a, r, s', \text{done})$ tuples up to `max_buffer_size=1000`.
  - `update_parameters()`: Samples batch, calculates target Q-values ($y = r + (1-d)\gamma (\min(Q_1', Q_2') - \alpha \log \pi)$), updates Critic loss via MSE, updates Actor loss ($J(\pi) = \mathbb{E}[\alpha \log \pi - Q]$), and performs soft Polyak target network updates ($\theta' \leftarrow \tau \theta + (1-\tau)\theta'$).
