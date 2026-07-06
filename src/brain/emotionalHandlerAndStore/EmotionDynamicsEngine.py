from __future__ import annotations
import os
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency availability is environment-specific.
    load_dotenv = None  # type: ignore[assignment]

try:
    from transformers import AutoModel
except ImportError:  # pragma: no cover - handled with a clear runtime error in load_model.
    AutoModel = None  # type: ignore[assignment]

try:
    from transferDTO import Latent, TokenBundle, Tokens
except ImportError:
    try:
        from ...transferDTO import Latent, TokenBundle, Tokens
    except ImportError:
        from src.transferDTO import Latent, TokenBundle, Tokens

try:
    from emotionalContract import AppraisalDTO, EmotionDTO, HomeostasisDTO
except ImportError:
    try:
        from .emotionalContract import AppraisalDTO, EmotionDTO, HomeostasisDTO
    except ImportError:
        from src.brain.emotionalHandlerAndStore.emotionalContract import (
            AppraisalDTO,
            EmotionDTO,
            HomeostasisDTO,
        )


if load_dotenv is not None:
    load_dotenv()


EmotionalStateVector = Dict[str, float]
EmotionTokenSequence = TokenBundle
EmotionalLatent = Latent

APPRAISAL_SCALARS: Tuple[str, ...] = (
    "novelty",
    "threat",
    "reward",
    "controllability",
    "urgency",
    "familiarity",
    "confidence",
    "prediction_error",
    "importance",
    "goal_relevance",
    "agency",
    "social_importance",
    "information_gain",
)

EMOTION_SCALARS: Tuple[str, ...] = (
    "joy",
    "sadness",
    "fear",
    "anger",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
    "curiosity",
    "confidence",
    "frustration",
    "motivation",
    "uncertainty",
    "emotional_intensity",
)

HOMEOSTASIS_SCALARS: Tuple[str, ...] = (
    "fatigue",
    "stress",
    "cognitive_load",
    "focus",
    "curiosity_drive",
    "novelty_hunger",
    "reward_satisfaction",
    "social_need",
    "stability_score",
)

DOMINANT_EMOTIONS: Tuple[str, ...] = (
    "joy",
    "sadness",
    "fear",
    "anger",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
    "curiosity",
    "confidence",
    "frustration",
    "motivation",
    "uncertainty",
)


def _env_bool(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"} #----------> 


def _env_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _env_float(name: str) -> float:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _env_str(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


# ==============================================================================
# EmotionInputBuilder
#
# Purpose
# Build deterministic scalar emotional state vectors for the dynamics model.
#
# Responsibilities
# Validate appraisal, previous emotion, and homeostasis DTOs; extract scalar fields;
# and produce a deterministic EmotionalStateVector with stable feature ordering.
#
# Inputs
# AppraisalDTO, previous EmotionDTO, HomeostasisDTO.
#
# Outputs
# EmotionalStateVector.
# ==============================================================================
class EmotionInputBuilder:

    def __init__(
        self,
        appraisal_fields: Sequence[str] = APPRAISAL_SCALARS,
        emotion_fields: Sequence[str] = EMOTION_SCALARS,
        homeostasis_fields: Sequence[str] = HOMEOSTASIS_SCALARS,
    ) -> None:
        self.appraisal_fields = tuple(appraisal_fields)
        self.emotion_fields = tuple(emotion_fields)
        self.homeostasis_fields = tuple(homeostasis_fields)

    def build(
        self,
        appraisal: AppraisalDTO,
        previous_emotion: EmotionDTO,
        homeostasis: HomeostasisDTO,
    ) -> EmotionalStateVector:
        self._validate_dto(appraisal, AppraisalDTO, "appraisal")
        self._validate_dto(previous_emotion, EmotionDTO, "previous_emotion")
        self._validate_dto(homeostasis, HomeostasisDTO, "homeostasis")

        state: EmotionalStateVector = {}
        state.update(self._extract_scalars("appraisal", appraisal, self.appraisal_fields))
        state.update(self._extract_scalars("previous_emotion", previous_emotion, self.emotion_fields))
        state.update(self._extract_scalars("homeostasis", homeostasis, self.homeostasis_fields))
        return state

    @property
    def feature_names(self) -> List[str]:
        return (
            [f"appraisal_{field}" for field in self.appraisal_fields]
            + [f"previous_emotion_{field}" for field in self.emotion_fields]
            + [f"homeostasis_{field}" for field in self.homeostasis_fields]
        )

    def _extract_scalars(self, prefix: str, dto: Any, fields: Iterable[str]) -> EmotionalStateVector:
        values: EmotionalStateVector = {}
        for field in fields:
            if not hasattr(dto, field):
                raise ValueError(f"{type(dto).__name__} is missing required field: {field}")
            raw_value = getattr(dto, field)
            if raw_value is None:
                raise ValueError(f"{type(dto).__name__}.{field} cannot be None")
            try:
                values[f"{prefix}_{field}"] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{type(dto).__name__}.{field} must be numeric") from exc
        return values

    def _validate_dto(self, value: Any, expected_type: type, argument_name: str) -> None:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{argument_name} must be {expected_type.__name__}; got {type(value).__name__}"
            )
        if not is_dataclass(value):
            raise TypeError(f"{argument_name} must be a dataclass DTO")


class EmotionEmbedding(nn.Module):

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names or EmotionInputBuilder().feature_names)
        self.embedding_dim = embedding_dim if embedding_dim is not None else _env_int("EMOTION_EMBEDDING_DIM")

        self.projections = nn.ModuleList([
            nn.Linear(1, self.embedding_dim) for _ in self.feature_names
        ])
        for proj in self.projections:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

    def embed(self, state_vector: EmotionalStateVector) -> EmotionTokenSequence:
        if not isinstance(state_vector, Mapping):
            raise TypeError(f"state_vector must be a mapping; got {type(state_vector).__name__}")

        values = []
        for name in self.feature_names:
            if name not in state_vector:
                raise ValueError(f"state_vector is missing required feature: {name}")
            try:
                values.append(float(state_vector[name]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"state_vector[{name!r}] must be numeric") from exc

        device = self.projections[0].weight.device
        dtype = self.projections[0].weight.dtype

        projected_tensors = []
        for i, val in enumerate(values):
            x_i = torch.tensor([[val]], dtype=dtype, device=device)
            projected_i = self.projections[i](x_i)
            projected_tensors.append(projected_i)

        projected = torch.stack(projected_tensors, dim=1)
        tokens = Tokens(values=projected, names=self.feature_names)
        return TokenBundle(groups={"emotional_state": tokens})

    def forward(self, state_vector: EmotionalStateVector) -> torch.Tensor:
        return self.embed(state_vector).tensor

class EmotionModel(nn.Module):

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: Optional[int] = None,
        fp16: Optional[bool] = None,
        bf16: Optional[bool] = None,
    ) -> None:

        super().__init__()
        self.model_name = model_name or _env_str("EMOTION_MODEL_NAME")
        self.device = torch.device(device or _env_str("EMOTION_DEVICE"))
        self.checkpoint_path = Path(checkpoint_path or _env_str("EMOTION_CHECKPOINT_PATH"))
        self.latent_dim = latent_dim if latent_dim is not None else _env_int("EMOTION_LATENT_DIM")
        self.fp16 = _env_bool("EMOTION_FP16", "false") if fp16 is None else fp16
        self.bf16 = _env_bool("EMOTION_BF16", "false") if bf16 is None else bf16
        self.backbone: Optional[nn.Module] = None
        self.input_projection: Optional[nn.Module] = None
        self.output_projection: Optional[nn.Module] = None
        self.hidden_state: Any = None

    def load_model(self) -> None:
        if self.backbone is not None:
            return
        if AutoModel is None:
            raise ImportError("transformers is required to load the EmotionModel Mamba backbone")

        dtype = self._torch_dtype()
        try:
            self.backbone = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
        except Exception as exc:
            raise RuntimeError(f"Failed to load emotion backbone model {self.model_name!r}") from exc

        hidden_size = self._resolve_hidden_size(self.backbone)
        embedding_dim = _env_int("EMOTION_EMBEDDING_DIM")
        self.input_projection = nn.Identity() if embedding_dim == hidden_size else nn.Linear(embedding_dim, hidden_size)
        self.output_projection = nn.Identity() if hidden_size == self.latent_dim else nn.Linear(hidden_size, self.latent_dim)
        super().to(self.device)

    def forward(self, token_sequence: EmotionTokenSequence) -> EmotionalLatent:
        if not isinstance(token_sequence, TokenBundle):
            raise TypeError(f"token_sequence must be TokenBundle; got {type(token_sequence).__name__}")
        self.load_model()
        assert self.backbone is not None
        assert self.input_projection is not None
        assert self.output_projection is not None

        x = token_sequence.tensor.to(self.device)
        x = x.to(dtype=self._runtime_dtype())
        inputs_embeds = self.input_projection(x)
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        sequence_output = self._extract_sequence_output(outputs)
        projected_sequence = self.output_projection(sequence_output)
        pooled = projected_sequence[:, -1, :]
        features = {
            name: projected_sequence[:, index, :]
            for index, name in enumerate(token_sequence.names)
            if index < projected_sequence.size(1)
        }
        return Latent(vector=pooled, features=features)

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        if self.backbone is None:
            raise RuntimeError("Cannot save EmotionModel checkpoint before load_model()")
        path = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.state_dict(),
                "latent_dim": self.latent_dim,
                "timestamp": datetime.utcnow().isoformat(),
            },
            path,
        )

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        path = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"EmotionModel checkpoint not found: {path}")
        self.load_model()
        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
            raise RuntimeError(f"Invalid EmotionModel checkpoint format: {path}")
        self.load_state_dict(checkpoint["state_dict"])

    def reset_hidden_state(self) -> None:
        self.hidden_state = None

    def to(self, *args: Any, **kwargs: Any) -> "EmotionModel":
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        return result

    def train(self, mode: bool = True) -> "EmotionModel":
        super().train(mode)
        return self

    def eval(self) -> "EmotionModel":
        return self.train(False)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def _torch_dtype(self) -> Optional[torch.dtype]:
        if self.fp16 and self.bf16:
            raise ValueError("Only one of EMOTION_FP16 or EMOTION_BF16 can be enabled")
        if self.fp16:
            return torch.float16
        if self.bf16:
            return torch.bfloat16
        return None

    def _runtime_dtype(self) -> torch.dtype:
        dtype = self._torch_dtype()
        return dtype or torch.float32

    def _resolve_hidden_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attribute in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, attribute, None)
            if value is not None:
                return int(value)
        raise RuntimeError("Unable to resolve hidden size from emotion backbone configuration")

    def _extract_sequence_output(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        raise RuntimeError("Emotion backbone output does not contain a sequence tensor")
    
class EmotionGenerator(nn.Module):

    def __init__(
        self,
        latent_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        emotion_names: Sequence[str] = DOMINANT_EMOTIONS,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim if latent_dim is not None else _env_int("EMOTION_LATENT_DIM")
        self.hidden_dim = hidden_dim if hidden_dim is not None else _env_int("EMOTION_HIDDEN_DIM")
        self.dropout = dropout if dropout is not None else _env_float("EMOTION_DROPOUT")
        self.emotion_names = tuple(emotion_names)
        self.scalar_names = tuple(name for name in EMOTION_SCALARS if name != "emotional_intensity")
        output_dim = len(self.scalar_names) + 1 + len(self.emotion_names)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def generate(self, latent: EmotionalLatent) -> EmotionDTO:
        if not isinstance(latent, Latent):
            raise TypeError(f"latent must be Latent; got {type(latent).__name__}")
        if latent.vector.dim() != 2:
            raise ValueError("latent.vector must have shape (batch_size, latent_dim)")
        if latent.vector.size(-1) != self.latent_dim:
            raise ValueError(
                f"latent.vector has dimension {latent.vector.size(-1)}; expected {self.latent_dim}"
            )

        predictions = self.mlp(latent.vector)
        scalar_count = len(self.scalar_names)
        scalar_values = torch.sigmoid(predictions[:, :scalar_count])
        intensity = torch.sigmoid(predictions[:, scalar_count : scalar_count + 1])
        dominant_logits = predictions[:, scalar_count + 1 :]
        dominant_index = int(torch.argmax(dominant_logits[0]).item())
        scalars = scalar_values[0].detach().float().cpu().tolist()
        intensity_value = float(intensity[0, 0].detach().float().cpu().item())

        payload = {name: float(value) for name, value in zip(self.scalar_names, scalars)}
        payload["dominant_emotion"] = self.emotion_names[dominant_index]
        payload["emotional_intensity"] = intensity_value
        return EmotionDTO(**payload)

    def forward(self, latent: EmotionalLatent) -> torch.Tensor:
        if not isinstance(latent, Latent):
            raise TypeError(f"latent must be Latent; got {type(latent).__name__}")
        return self.mlp(latent.vector)


class EmotionDynamicsEngine(nn.Module):

    def __init__(
        self,
        input_builder: EmotionInputBuilder,
        embedding: EmotionEmbedding,
        model: EmotionModel,
        generator: EmotionGenerator,
    ) -> None:
        super().__init__()
        if not isinstance(input_builder, EmotionInputBuilder):
            raise TypeError("input_builder must be an EmotionInputBuilder")
        if not isinstance(embedding, EmotionEmbedding):
            raise TypeError("embedding must be an EmotionEmbedding")
        if not isinstance(model, EmotionModel):
            raise TypeError("model must be an EmotionModel")
        if not isinstance(generator, EmotionGenerator):
            raise TypeError("generator must be an EmotionGenerator")
        self.input_builder = input_builder
        self.embedding = embedding
        self.model = model
        self.generator = generator

    def evaluate(
        self,
        appraisal: AppraisalDTO,
        previous_emotion: EmotionDTO,
        homeostasis: HomeostasisDTO,
    ) -> EmotionDTO:

        with torch.no_grad():
            return self.forward(appraisal, previous_emotion, homeostasis)

    def forward(
        self,
        appraisal: AppraisalDTO,
        previous_emotion: EmotionDTO,
        homeostasis: HomeostasisDTO,
    ) -> EmotionDTO:
        state_vector = self.input_builder.build(appraisal, previous_emotion, homeostasis)
        token_sequence = self.embedding.embed(state_vector)
        latent = self.model.forward(token_sequence)
        return self.generator.generate(latent)


__all__ = [
    "EmotionalStateVector",
    "EmotionTokenSequence",
    "EmotionalLatent",
    "EmotionInputBuilder",
    "EmotionEmbedding",
    "EmotionModel",
    "EmotionGenerator",
    "EmotionDynamicsEngine",
]
