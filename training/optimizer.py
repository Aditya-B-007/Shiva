import numpy as np
from typing import Dict, List, Tuple


class AdamW:
    def __init__(
        self,
        params: List[Tuple[str, np.ndarray]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        self.params = params  
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.step_count = 0
        self.m: Dict[str, np.ndarray] = {name: np.zeros_like(p) for name, p in params}
        self.v: Dict[str, np.ndarray] = {name: np.zeros_like(p) for name, p in params}

    def clip_gradients(self, grads: Dict[str, np.ndarray]) -> None:
        total_norm_sq = 0.0
        for name, g in grads.items():
            total_norm_sq += float(np.sum(g ** 2))
        total_norm = np.sqrt(total_norm_sq)

        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-12)
            for name in grads:
                grads[name] *= scale

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.step_count += 1
        self.clip_gradients(grads)

        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)

        for name, param in self.params:
            if name not in grads:
                continue

            g = grads[name]

            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2
            
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)