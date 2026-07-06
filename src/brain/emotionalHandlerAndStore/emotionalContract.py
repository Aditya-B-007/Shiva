from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Generic,
    TypeVar,
)
from uuid import uuid4 #For debugging purpose

T = TypeVar("T")

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

    audio: Optional[Any] = None

    image: Optional[Any] = None

    video: Optional[Any] = None

    sensor_data: Dict[str, Any] = field(default_factory=dict)

    text_embedding: Optional[Any] = None

    vision_embedding: Optional[Any] = None

    audio_embedding: Optional[Any] = None

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

    #=============Internal Cognitive state DTOs=============

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

#==========Appraisal related DTOs===================

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

