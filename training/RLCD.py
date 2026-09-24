import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .optimizer import AdamW


@dataclass
class RLCDSample:
    z_positive: np.ndarray      # Calibrated/rubric-conditioned vector: [1, 512]
    z_negative: np.ndarray      # Raw/unconstrained vector: [1, 512]
    options_matrix: np.ndarray  # Evaluated candidate actions: [K, 512]


class RLCDTrainer:
    def __init__(self, temperature: float = 0.07, lr: float = 5e-4):
        self.temperature = temperature
        self.W_cal = np.eye(512, dtype=np.float32)
        self.b_cal = np.zeros(512, dtype=np.float32)
        params = [("W_cal", self.W_cal), ("b_cal", self.b_cal)]
        self.optimizer = AdamW(params, lr=lr, weight_decay=0.001)

    def _calibrate_vector(self, z: np.ndarray) -> np.ndarray:
        return np.matmul(z, self.W_cal) + self.b_cal

    def train_step(self, sample: RLCDSample) -> float:
        M = sample.options_matrix  # [K, 512]

        # 1. Forward Student Distribution
        z_cal = self._calibrate_vector(sample.z_negative)  # [1, 512]
        eps = 1e-12
        z_cal_norm = z_cal / (np.linalg.norm(z_cal, ord=2, axis=-1, keepdims=True) + eps)

        student_logits = np.matmul(z_cal_norm, M.T) / self.temperature
        student_exp = np.exp(student_logits - np.max(student_logits, axis=-1, keepdims=True))
        student_probs = student_exp / np.sum(student_exp, axis=-1, keepdims=True)  # [1, K]

        # 2. Construct Supervised Teacher Distribution
        z_pos_norm = sample.z_positive / (np.linalg.norm(sample.z_positive, ord=2, axis=-1, keepdims=True) + eps)
        teacher_logits = np.matmul(z_pos_norm, M.T) / self.temperature

        teacher_exp = np.exp(teacher_logits - np.max(teacher_logits, axis=-1, keepdims=True))
        teacher_probs = teacher_exp / np.sum(teacher_exp, axis=-1, keepdims=True)  # [1, K]

        # 3. Distillation Cross-Entropy Loss
        loss = -float(np.sum(teacher_probs * np.log(student_probs + eps)))

        # 4. Analytical Gradients w.r.t Calibration Parameters
        # dL / d(student_logits) = student_probs - teacher_probs
        grad_logits = (student_probs - teacher_probs) / self.temperature  # [1, K]
        grad_z_cal = np.matmul(grad_logits, M)                            # [1, 512]

        grad_W_cal = np.matmul(sample.z_negative.T, grad_z_cal)           # [512, 512]
        grad_b_cal = np.sum(grad_z_cal, axis=0)                           # [512]

        # 5. Optimizer step
        self.optimizer.step({"W_cal": grad_W_cal, "b_cal": grad_b_cal})
        return loss