from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")

# ==============================================================================
# SECTION 1: Core Token & Latent Representation DTOs
# ==============================================================================

@dataclass(slots=True)
class Tokens:
    """Encapsulates a group of tokens and their associated feature names."""
    values: torch.Tensor  # shape: (batch_size, num_features, vector_size)
    names: List[str]      # names of the features in this group


@dataclass(slots=True)
class TokenBundle:
    """Encapsulates multiple Token groups and provides utility to merge them."""
    groups: Dict[str, Tokens]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def tensor(self) -> torch.Tensor:
        tensors = [g.values for g in self.groups.values() if g.values.size(1) > 0]
        if not tensors:
            raise ValueError("No active tokens found in TokenBundle")
        return torch.cat(tensors, dim=1)

    @property
    def names(self) -> List[str]:
        feature_names = []
        for g in self.groups.values():
            feature_names.extend(g.names)
        return feature_names


@dataclass(slots=True)
class Latent:
    """Carries the pooled latent state vector and contextualized representations."""
    vector: torch.Tensor            # Pooled representation (e.g., shape: (batch_size, vector_size))
    features: Dict[str, torch.Tensor]  # Maps each feature name to its post-attention encoded tensor
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ==============================================================================
# SECTION 2: Perception & Observation DTOs
# ==============================================================================

class ObservationKind(str, Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PerceptionObservationDTO:
    device: str
    kind: ObservationKind
    summary: str
    payload: Any = None
    payload_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class PerceptionBundleDTO:
    query: str
    observations: List[PerceptionObservationDTO] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_observations(self) -> bool:
        return bool(self.observations)

    def observation_names(self) -> List[str]:
        return [observation.device for observation in self.observations]

    def errors(self) -> List[PerceptionObservationDTO]:
        return [
            observation
            for observation in self.observations
            if observation.kind == ObservationKind.ERROR
        ]


@dataclass(frozen=True, slots=True)
class PerceptionCaptureRequestDTO:
    device: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None


# ==============================================================================
# SECTION 3: Memory Node & Graph DTOs
# ==============================================================================

@dataclass(frozen=True, slots=True)
class MemoryNodeDTO:
    id: str
    raw_content: Any
    summary: str
    modality: str
    semantic_type: str
    activation: float
    strength: float
    emotional_salience: float
    identity_relevance: float
    recency: float
    status: str
    access_count: int
    created_at: datetime
    last_accessed: datetime | None = None
    context_signature: str = ""


@dataclass(frozen=True, slots=True)
class MemoryEdgeDTO:
    source: str
    destination: str
    association_strength: float
    association_type: str
    activation_probability: float
    traversal_count: int
    creation_time: datetime
    last_traversed: datetime | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryGraphDTO:
    nodes: tuple[MemoryNodeDTO, ...] = field(default_factory=tuple)
    edges: tuple[MemoryEdgeDTO, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RetrievalDTO:
    query: object
    memories: tuple[MemoryNodeDTO, ...] = field(default_factory=tuple)
    confidence: float = 0.0


# ==============================================================================
# SECTION 4: Cognitive Brain & Reasoning DTOs
# ==============================================================================

@dataclass(slots=True)
class BrainErrorDTO:
    component: str
    message: str
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ThoughtParseDiagnosticsDTO:
    parse_success: bool = True
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ThoughtDTO:
    raw_text: str
    thought_body: str
    critique: str
    confidence: float
    parsed_decision: Optional[str] = None
    parse_diagnostics: ThoughtParseDiagnosticsDTO = field(default_factory=ThoughtParseDiagnosticsDTO)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ReasoningContextDTO:
    perception: Any
    memories: List[Any] = field(default_factory=list)
    emotion: Optional[Any] = None
    thoughts: List[ThoughtDTO] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class NodeReasoningResultDTO:
    decision: Optional[str]
    confidence: float
    thought_history: List[ThoughtDTO]
    iterations_used: int
    max_iterations: int
    goal_reached: bool
    errors: List[BrainErrorDTO] = field(default_factory=list)
    timed_out: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ColumnResultDTO:
    column_id: int
    role: str
    result: Optional[NodeReasoningResultDTO] = None
    error: Optional[BrainErrorDTO] = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class MothershipResponseDTO:
    decision: Optional[str]
    confidence: float
    goal_reached: bool
    observations_used: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    cycles_used: int = 0
    errors: List[BrainErrorDTO] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class BrainHealthDTO:
    status: str
    cognitive_core_online: bool
    decoder_online: bool
    memory_online: bool
    homeostasis_online: bool
    errors: List[BrainErrorDTO] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class EncoderInputDTO:
    text: Union[str, List[str]]
    max_length: int = 512


@dataclass(slots=True)
class EncoderOutputDTO:
    last_hidden_state_pt: torch.Tensor
    last_hidden_state_np: np.ndarray
    pooler_output_pt: Optional[torch.Tensor] = None
    pooler_output_np: Optional[np.ndarray] = None


@dataclass(slots=True)
class DecoderInputDTO:
    vector_input: Union[torch.Tensor, np.ndarray]


@dataclass(slots=True)
class DecoderOutputDTO:
    logits_pt: torch.Tensor
    logits_np: np.ndarray
    hidden_states_pt: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
    hidden_states_np: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None


# ==============================================================================
# SECTION 5: Emotional Dynamics & Appraisal DTOs
# ==============================================================================

class EventType(Enum):
    PERCEPTION = auto()
    USER_INTERACTION = auto()
    SYSTEM_INTERACTION = auto()
    SENSOR_UPDATE = auto()
    ENVIRONMENT_UPDATE = auto()
    MEMORY_RETRIEVAL = auto()
    MEMORY_STORAGE = auto()
    MEMORY_FORGET = auto()
    GOAL_CREATED = auto()
    GOAL_COMPLETED = auto()
    GOAL_FAILED = auto()
    GOAL_CANCELLED = auto()
    ACTION_STARTED = auto()
    ACTION_COMPLETED = auto()
    ACTION_FAILED = auto()
    TOOL_EXECUTION = auto()
    TOOL_RESULT = auto()
    EMOTION_UPDATE = auto()
    HOMEOSTASIS_UPDATE = auto()
    IDENTITY_UPDATE = auto()
    STARTUP = auto()
    SHUTDOWN = auto()
    SLEEP = auto()
    WAKE = auto()


@dataclass(slots=True)
class Event(Generic[T]):
    event_type: EventType
    payload: T
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class PerceptionDTO:
    text: Optional[str] = None
    language: Optional[str] = None
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    text_embedding: Optional[Any] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class EnvironmentDTO:
    battery_percentage: float = 100.0
    charging: bool = False
    internet_available: bool = True
    device_awake: bool = True
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    available_memory: float = 0.0
    current_time: datetime = field(default_factory=datetime.utcnow)
    location: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoalDTO:
    goal_id: str
    description: str
    priority: float = 0.5
    progress: float = 0.0
    completed: bool = False
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class EmotionDTO:
    joy: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0
    curiosity: float = 0.5
    confidence: float = 0.5
    frustration: float = 0.0
    motivation: float = 0.5
    uncertainty: float = 0.0
    dominant_emotion: Optional[str] = None
    emotional_intensity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class HomeostasisDTO:
    fatigue: float = 0.0
    stress: float = 0.0
    cognitive_load: float = 0.0
    focus: float = 1.0
    curiosity_drive: float = 0.5
    novelty_hunger: float = 0.5
    reward_satisfaction: float = 0.5
    social_need: float = 0.5
    stability_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class IdentityDTO:
    identity_id: str
    name: str = "Shiva"
    beliefs: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    active_goals: List[GoalDTO] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class MemoryDTO:
    episodic_memory: List[Any] = field(default_factory=list)
    semantic_memory: List[Any] = field(default_factory=list)
    working_memory: List[Any] = field(default_factory=list)
    retrieved_memory: Optional[Any] = None
    retrieval_confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class AppraisalDTO:
    novelty: float = 0.0
    threat: float = 0.0
    reward: float = 0.0
    controllability: float = 0.0
    urgency: float = 0.0
    familiarity: float = 0.0
    confidence: float = 0.0
    prediction_error: float = 0.0
    importance: float = 0.0
    goal_relevance: float = 0.0
    agency: float = 0.0
    social_importance: float = 0.0
    information_gain: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class FeatureBundle:
    event: Event
    perception: Optional[PerceptionDTO] = None
    environment: Optional[EnvironmentDTO] = None
    emotion: Optional[EmotionDTO] = None
    homeostasis: Optional[HomeostasisDTO] = None
    identity: Optional[IdentityDTO] = None
    memory: Optional[MemoryDTO] = None
    latent_state: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    bundle_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class NumericalFeatureVector:
    numerical_features: Dict[str, float]
    categorical_features: Dict[str, str]
    embeddings: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class FeatureTokenSequence:
    tokens: Any
    feature_names: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class CognitiveLatent:
    latent_vector: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
