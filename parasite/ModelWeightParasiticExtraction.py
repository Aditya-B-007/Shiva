from __future__ import annotations
import copy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from core.interfaces import IAlignmentLoss, IRepresentationProbe


# ---------------------------------------------------------------------------
# Activation buffer: stores the most recent forward-hook capture
# ---------------------------------------------------------------------------

class ActivationBuffer:
    def __init__(self) -> None:
        self._activation: Optional[torch.Tensor] = None

    def capture(self, value: torch.Tensor) -> None:
        # Detach: we never want gradients to flow back through the host.
        self._activation = value.detach()

    def read(self) -> torch.Tensor:
        if self._activation is None:
            raise RuntimeError(
                "ActivationBuffer is empty. "
                "Run a forward pass through the host model before calling read()."
            )
        return self._activation

    def clear(self) -> None:
        self._activation = None


# ---------------------------------------------------------------------------
# ProbeNetwork: projects host activations into Shiva's latent space
# ---------------------------------------------------------------------------

class ProbeNetwork(nn.Module):
    def __init__(self, host_dim: int, target_dim: int) -> None:
        super().__init__()
        hidden = max(host_dim, target_dim)
        self.net = nn.Sequential(
            nn.Linear(host_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            h = h.mean(dim=1)   # sequence pooling before projection
        return self.net(h)


# ---------------------------------------------------------------------------
# ParasiticExtractor: IRepresentationProbe implementation
# ---------------------------------------------------------------------------

class ParasiticExtractor(IRepresentationProbe, nn.Module):
    def __init__(
        self,
        host_dim: int,
        target_dim: int,
        loss_fn: Optional[IAlignmentLoss] = None,
        lr: float = 3e-4,
        ema_momentum: float = 0.99,
        use_ema: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.host_dim = host_dim
        self.target_dim = target_dim
        self.ema_momentum = ema_momentum
        self.use_ema = use_ema

        self.probe = ProbeNetwork(host_dim, target_dim)
        self.optimizer = torch.optim.AdamW(self.probe.parameters(), lr=lr)

        if use_ema:
            self.target_probe = copy.deepcopy(self.probe)
            for p in self.target_probe.parameters():
                p.requires_grad_(False)
        else:
            self.target_probe = None

        # Import here to avoid circular dependency if IAlignmentLoss is
        # defined in latent_alignment.py.  We import only the concrete
        # default; callers may inject any IAlignmentLoss.
        if loss_fn is None:
            from core.latent_alignment import InfoNCELoss
            loss_fn = InfoNCELoss(temperature=0.07)
        self._loss_fn = loss_fn

        # Hook state
        self._buffer = ActivationBuffer()
        self._hooks: List[RemovableHandle] = []

    # ------------------------------------------------------------------
    # IRepresentationProbe: hook management
    # ------------------------------------------------------------------

    def attach(self, host_model: nn.Module, layer_name: str) -> None:
        # Resolve the target sub-module.
        target_layer = self._resolve_layer(host_model, layer_name)

        def _hook(module, input, output):  # noqa: ANN001
            # output may be a tuple (e.g. transformer layers return (hidden, attn)).
            activation = output[0] if isinstance(output, (tuple, list)) else output
            self._buffer.capture(activation)

        handle = target_layer.register_forward_hook(_hook)
        self._hooks.append(handle)

    def detach(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._buffer.clear()

    # ------------------------------------------------------------------
    # IRepresentationProbe: distillation step
    # ------------------------------------------------------------------

    def distil_step(
        self,
        host_input: torch.Tensor,
        target_encoder: nn.Module,
    ) -> float:
        h_host = self._buffer.read()
        with torch.no_grad():
            z_shiva = target_encoder.forward_pass(host_input).mean(dim=1)
        z_probe = self.probe(h_host)
        if self.use_ema and self.target_probe is not None:
            with torch.no_grad():
                z_positive = self.target_probe(h_host)
            loss = self._loss_fn.compute(z_probe, z_positive)
        else:
            loss = self._loss_fn.compute(z_probe, z_shiva)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_ema and self.target_probe is not None:
            self._ema_update()

        self._buffer.clear()
        return loss.item()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _ema_update(self) -> None:
        m = self.ema_momentum
        for p_online, p_target in zip(
            self.probe.parameters(), self.target_probe.parameters()  # type: ignore[union-attr]
        ):
            p_target.data = m * p_target.data + (1 - m) * p_online.data

    @staticmethod
    def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
        parts = layer_name.split(".")
        current: nn.Module = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise AttributeError(
                    f"Layer '{part}' not found on {type(current).__name__}. "
                    f"Full path attempted: '{layer_name}'.\n"
                    f"Available sub-modules: {list(current._modules.keys())}"
                )
        return current

    # ------------------------------------------------------------------
    # Convenience context manager
    # ------------------------------------------------------------------

    def probe_context(self, host_model: nn.Module, layer_name: str):
        class _Ctx:
            def __init__(self_, extractor, model, name):
                self_._extractor = extractor
                self_._model = model
                self_._name = name

            def __enter__(self_):
                self_._extractor.attach(self_._model, self_._name)
                return self_._extractor

            def __exit__(self_, *args):
                self_._extractor.detach()

        return _Ctx(self, host_model, layer_name)


    
    def compute_loss(self,host_input,target_encoder):
        h_host=self._buffer.read()
        with torch.no_grad():
            z_shiva=(
                target_encoder
                .forward_pass(host_input)
                .mean(dim=1)
            )
            z_probe=self.probe(h_host)

            return self._loss_fn.compute(
                z_probe,
                z_shiva
            )
