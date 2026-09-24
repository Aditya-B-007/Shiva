import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.transformerAndRL.transformer import FinalMLP, silu
from src.output.optionMatrixConvertor import OptionMatrixConvertor
from .optimizer import AdamW


@dataclass
class DPOSample:
    context_vector: np.ndarray        
    action_strings: List[str]       
    winner_idx: int               
    loser_idx: int                 
    ref_logits: np.ndarray


class DPOTrainer:
    def __init__(
        self,
        final_mlp: FinalMLP,
        option_convertor: OptionMatrixConvertor,
        beta: float = 0.1,
        temperature: float = 0.07,
        lr: float = 1e-4
    ):
        self.mlp = final_mlp
        self.convertor = option_convertor
        self.beta = beta
        self.temperature = temperature

        params = [
            ("mlp_W_gate", self.mlp.W_gate),
            ("mlp_b_gate", self.mlp.b_gate),
            ("mlp_W_up", self.mlp.W_up),
            ("mlp_b_up", self.mlp.b_up),
            ("mlp_W_down", self.mlp.W_down),
            ("mlp_b_down", self.mlp.b_down),
            ("opt_W_proj", self.convertor.W_proj),
            ("opt_b_proj", self.convertor.b_proj),
        ]
        self.optimizer = AdamW(params, lr=lr)

    def _backward_final_mlp(self, x: np.ndarray, grad_z: np.ndarray) -> Dict[str, np.ndarray]:

        gate_lin = np.matmul(x, self.mlp.W_gate) + self.mlp.b_gate
        sig = 1.0 / (1.0 + np.exp(-np.clip(gate_lin, -30.0, 30.0)))
        gate_act = gate_lin * sig  # silu(gate_lin)

        up_lin = np.matmul(x, self.mlp.W_up) + self.mlp.b_up
        h_mid = gate_act * up_lin  # [1, 512]

        grad_W_down = np.matmul(h_mid.T, grad_z)   # [512, 512]
        grad_b_down = np.sum(grad_z, axis=0)       # [512]
        grad_h_mid = np.matmul(grad_z, self.mlp.W_down.T)  # [1, 512]

        grad_up_lin = grad_h_mid * gate_act
        grad_gate_act = grad_h_mid * up_lin

        dsilu = sig + gate_lin * sig * (1.0 - sig)
        grad_gate_lin = grad_gate_act * dsilu

        grad_W_gate = np.matmul(x.T, grad_gate_lin)
        grad_b_gate = np.sum(grad_gate_lin, axis=0)

        grad_W_up = np.matmul(x.T, grad_up_lin)
        grad_b_up = np.sum(grad_up_lin, axis=0)

        return {
            "mlp_W_gate": grad_W_gate,
            "mlp_b_gate": grad_b_gate,
            "mlp_W_up": grad_W_up,
            "mlp_b_up": grad_b_up,
            "mlp_W_down": grad_W_down,
            "mlp_b_down": grad_b_down,
        }

    def train_step(self, sample: DPOSample) -> Tuple[float, float]:

        x = sample.context_vector
        w = sample.winner_idx
        l = sample.loser_idx

        z = self.mlp.forward(x)
        eps = 1e-12
        z_norm = z / (np.linalg.norm(z, ord=2, axis=-1, keepdims=True) + eps)

        M = self.convertor.convert(sample.action_strings)

        logits = np.matmul(z_norm, M.T) / self.temperature  # [1, K]
        pi_margin = float(logits[0, w] - logits[0, l])
        ref_margin = float(sample.ref_logits[0, w] - sample.ref_logits[0, l])

        h = self.beta * (pi_margin - ref_margin)
        loss = float(np.log1p(np.exp(-h))) if h >= 0 else float(-h + np.log1p(np.exp(h)))

        sigmoid_neg_h = 1.0 / (1.0 + np.exp(np.clip(h, -30.0, 30.0)))
        scale = (self.beta / self.temperature) * sigmoid_neg_h
        # L = -log sigmoid(h). dL/dh = -sigmoid(-h).
        # dh/dz = (beta/T) * (M[w] - M[l])
        # dL/dz = -scale * (M[w:w+1] - M[l:l+1]) = scale * (M[l:l+1] - M[w:w+1])
        # BUT for gradient descent param -= lr * grad, grad must be dL/d(param).
        # dL/dz_norm = -scale * (M[w] - M[l]) = scale * (M[l] - M[w])
        grad_z = scale * (M[l:l+1] - M[w:w+1])  # [1, 512]

        # dL/dM:
        # dh/dM[w] = (beta/T) * z_norm  => dL/dM[w] = -scale * z_norm
        # dh/dM[l] = -(beta/T) * z_norm => dL/dM[l] = +scale * z_norm
        grad_M = np.zeros_like(M)
        grad_M[w:w+1] = -scale * z_norm
        grad_M[l:l+1] = +scale * z_norm

        mlp_grads = self._backward_final_mlp(x, grad_z)

        # Backward through L2 normalization on options matrix:
        # M = P / ||P||. For y = p / ||p||:
        # dy/dp = (I - y y^T) / ||p||
        if hasattr(self.convertor, "_last_pooled") and hasattr(self.convertor, "_last_projected"):
            P = self.convertor._last_projected  # [K, 512]
            norms = self.convertor._last_norms  # [K, 1]
            # grad_P = (grad_M - M * sum(grad_M * M, axis=-1, keepdims=True)) / norms
            proj_grad = (grad_M - M * np.sum(grad_M * M, axis=-1, keepdims=True)) / np.maximum(norms, eps)
            grad_W_proj = np.matmul(self.convertor._last_pooled.T, proj_grad)  # [768, 512]
            grad_b_proj = np.sum(proj_grad, axis=0)                            # [512]
        else:
            grad_W_proj = np.zeros_like(self.convertor.W_proj)
            grad_b_proj = np.zeros_like(self.convertor.b_proj)

        all_grads = {
            **mlp_grads,
            "opt_W_proj": grad_W_proj,
            "opt_b_proj": grad_b_proj
        }

        self.optimizer.step(all_grads)
        return loss, h