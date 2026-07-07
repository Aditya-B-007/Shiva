from __future__ import annotations
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil

# Setup robust imports to handle relative and absolute imports in python paths
try:
    from emotionInterface import IAppraisal, IFeatureEmbedding
    from emotionalContract import (
        EventType,
        FeatureBundle,
        NumericalFeatureVector,
        AppraisalDTO
    )
except ImportError:
    try:
        from .emotionInterface import IAppraisal, IFeatureEmbedding
        from .emotionalContract import (
            EventType,
            FeatureBundle,
            NumericalFeatureVector,
            AppraisalDTO
        )
    except ImportError:
        from src.brain.emotionalHandlerAndStore.emotionInterface import IAppraisal, IFeatureEmbedding
        from src.brain.emotionalHandlerAndStore.emotionalContract import (
            EventType,
            FeatureBundle,
            NumericalFeatureVector,
            AppraisalDTO
        )

# Import simplified generic DTOs from transferDTO
try:
    from transferDTO import Tokens, TokenBundle, Latent
except ImportError:
    try:
        from ...transferDTO import Tokens, TokenBundle, Latent
    except ImportError:
        from src.transferDTO import Tokens, TokenBundle, Latent

try:
    from transformerArchitecture import TransformerConfig, Encoder
except ImportError:
    try:
        from ..transformerArchitecture import TransformerConfig, Encoder
    except ImportError:
        from src.brain.transformerArchitecture import TransformerConfig, Encoder


#============CONSTANTS===============
NUMBERS = [
    # Perception
    "perception_confidence",
    # Environment
    "env_battery_percentage",
    "env_cpu_utilization",
    "env_gpu_utilization",
    "env_available_memory",
    # Emotion
    "joy", "sadness", "fear", "anger", "surprise", "disgust", "trust", 
    "anticipation", "curiosity", "emotion_confidence", "frustration", 
    "motivation", "uncertainty", "emotional_intensity",
    # Homeostasis
    "fatigue", "stress", "cognitive_load", "focus", "curiosity_drive", 
    "novelty_hunger", "reward_satisfaction", "social_need", "stability_score",
    # Memory
    "memory_retrieval_confidence",
]

CATEGORIES = [
    "event_type",
    "event_source",
    "env_charging",
    "env_internet_available",
    "env_device_awake",
    "dominant_emotion",
]

# Fetch EventType names for vocabulary building
try:
    EVENTS = [t.name for t in EventType]
except Exception:
    EVENTS = [
        "PERCEPTION", "USER_INTERACTION", "SYSTEM_INTERACTION", "SENSOR_UPDATE",
        "ENVIRONMENT_UPDATE", "MEMORY_RETRIEVAL", "MEMORY_STORAGE", "MEMORY_FORGET",
        "GOAL_CREATED", "GOAL_COMPLETED", "GOAL_FAILED", "GOAL_CANCELLED",
        "ACTION_STARTED", "ACTION_COMPLETED", "ACTION_FAILED", "TOOL_EXECUTION",
        "TOOL_RESULT", "EMOTION_UPDATE", "HOMEOSTASIS_UPDATE", "IDENTITY_UPDATE",
        "STARTUP", "SHUTDOWN", "SLEEP", "WAKE"
    ]

VOCABS = {
    "event_type": ["UNK"] + EVENTS,
    "event_source": ["UNK", "user", "system", "perception", "sensor", "environment", "memory", "goal", "action", "tool", "emotion", "homeostasis", "identity"],
    "env_charging": ["UNK", "True", "False", "None"],
    "env_internet_available": ["UNK", "True", "False", "None"],
    "env_device_awake": ["UNK", "True", "False", "None"],
    "dominant_emotion": ["UNK", "joy", "sadness", "fear", "anger", "surprise", "disgust", "trust", "anticipation", "curiosity", "confidence", "frustration", "motivation", "uncertainty", "None"]
}


#============HELPER FUNCTIONS===============


#============ABSTRACT CLASSES AND PROTOCOLS===============


#============CONCRETE IMPLEMENTATIONS & DATA STRUCTURES===============
class FeatureExtractor:
    """
    FeatureExtractor extracts numerical values, categorical values, and pre-existing
    embeddings from a FeatureBundle.
    """

    def _read_dtos(self, bundle: FeatureBundle) -> Dict[str, Any]:
        return {
            "event": bundle.event,
            "perception": bundle.perception,
            "environment": bundle.environment,
            "emotion": bundle.emotion,
            "homeostasis": bundle.homeostasis,
            "identity": bundle.identity,
            "memory": bundle.memory
        }

    def _extract_numerical(self, dtos: Dict[str, Any]) -> Dict[str, float]:
        """Extracts numerical values from the respective DTOs."""
        numerical = {}
        
        # Perception
        p = dtos.get("perception")
        numerical["perception_confidence"] = float(p.confidence) if p and p.confidence is not None else 1.0
        
        # Environment
        env = dtos.get("environment")
        numerical["env_battery_percentage"] = float(env.battery_percentage) if env and env.battery_percentage is not None else 100.0
        
        # CPU Utilization
        if env and env.cpu_utilization is not None and env.cpu_utilization > 0.0:
            numerical["env_cpu_utilization"] = float(env.cpu_utilization)
        else:
            try:
                import psutil
                numerical["env_cpu_utilization"] = float(psutil.cpu_percent())
            except Exception:
                numerical["env_cpu_utilization"] = float(env.cpu_utilization) if env and env.cpu_utilization is not None else 0.0
                
        # GPU Utilization (Default to DTO value if present)
        numerical["env_gpu_utilization"] = float(env.gpu_utilization) if env and env.gpu_utilization is not None else 0.0
        
        # Available Memory (in GB)
        if env and env.available_memory is not None and env.available_memory > 0.0:
            numerical["env_available_memory"] = float(env.available_memory)
        else:
            try:
                import psutil
                # Convert bytes to GB
                numerical["env_available_memory"] = float(psutil.virtual_memory().available / (1024 ** 3))
            except Exception:
                numerical["env_available_memory"] = float(env.available_memory) if env and env.available_memory is not None else 0.0
        
        # Emotion
        emo = dtos.get("emotion")
        numerical["joy"] = float(emo.joy) if emo and emo.joy is not None else 0.0
        numerical["sadness"] = float(emo.sadness) if emo and emo.sadness is not None else 0.0
        numerical["fear"] = float(emo.fear) if emo and emo.fear is not None else 0.0
        numerical["anger"] = float(emo.anger) if emo and emo.anger is not None else 0.0
        numerical["surprise"] = float(emo.surprise) if emo and emo.surprise is not None else 0.0
        numerical["disgust"] = float(emo.disgust) if emo and emo.disgust is not None else 0.0
        numerical["trust"] = float(emo.trust) if emo and emo.trust is not None else 0.0
        numerical["anticipation"] = float(emo.anticipation) if emo and emo.anticipation is not None else 0.0
        numerical["curiosity"] = float(emo.curiosity) if emo and emo.curiosity is not None else 0.5
        numerical["emotion_confidence"] = float(emo.confidence) if emo and emo.confidence is not None else 0.5
        numerical["frustration"] = float(emo.frustration) if emo and emo.frustration is not None else 0.0
        numerical["motivation"] = float(emo.motivation) if emo and emo.motivation is not None else 0.5
        numerical["uncertainty"] = float(emo.uncertainty) if emo and emo.uncertainty is not None else 0.0
        numerical["emotional_intensity"] = float(emo.emotional_intensity) if emo and emo.emotional_intensity is not None else 0.0
        
        # Homeostasis
        h = dtos.get("homeostasis")
        numerical["fatigue"] = float(h.fatigue) if h and h.fatigue is not None else 0.0
        numerical["stress"] = float(h.stress) if h and h.stress is not None else 0.0
        numerical["cognitive_load"] = float(h.cognitive_load) if h and h.cognitive_load is not None else 0.0
        numerical["focus"] = float(h.focus) if h and h.focus is not None else 1.0
        numerical["curiosity_drive"] = float(h.curiosity_drive) if h and h.curiosity_drive is not None else 0.5
        numerical["novelty_hunger"] = float(h.novelty_hunger) if h and h.novelty_hunger is not None else 0.5
        numerical["reward_satisfaction"] = float(h.reward_satisfaction) if h and h.reward_satisfaction is not None else 0.5
        numerical["social_need"] = float(h.social_need) if h and h.social_need is not None else 0.5
        numerical["stability_score"] = float(h.stability_score) if h and h.stability_score is not None else 1.0
        
        # Memory
        m = dtos.get("memory")
        numerical["memory_retrieval_confidence"] = float(m.retrieval_confidence) if m and m.retrieval_confidence is not None else 1.0
        
        return numerical

    def _extract_categorical(self, dtos: Dict[str, Any]) -> Dict[str, str]:
        """Extracts categorical values from the respective DTOs."""
        categorical = {}
        
        # Event
        evt = dtos.get("event")
        categorical["event_type"] = evt.event_type.name if evt and evt.event_type is not None else "UNK"
        categorical["event_source"] = str(evt.source) if evt and evt.source is not None else "UNK"
        
        # Environment
        env = dtos.get("environment")
        categorical["env_charging"] = str(env.charging) if env and env.charging is not None else "None"
        categorical["env_internet_available"] = str(env.internet_available) if env and env.internet_available is not None else "None"
        categorical["env_device_awake"] = str(env.device_awake) if env and env.device_awake is not None else "None"
        
        # Emotion
        emo = dtos.get("emotion")
        categorical["dominant_emotion"] = str(emo.dominant_emotion) if emo and emo.dominant_emotion is not None else "None"
        
        return categorical

    def _extract_embeddings(self, dtos: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts embeddings if already available in PerceptionDTO."""
        embeddings = {}
        p = dtos.get("perception")
        if p:
            if p.text_embedding is not None:
                embeddings["text_embedding"] = p.text_embedding
            if p.vision_embedding is not None:
                embeddings["vision_embedding"] = p.vision_embedding
            if p.audio_embedding is not None:
                embeddings["audio_embedding"] = p.audio_embedding
        return embeddings

    def _produce_vector(
        self,
        numerical: Dict[str, float],
        categorical: Dict[str, str],
        embeddings: Dict[str, Any]
    ) -> NumericalFeatureVector:
        """Produces a structured feature vector container."""
        return NumericalFeatureVector(
            numerical_features=numerical,
            categorical_features=categorical,
            embeddings=embeddings
        )

    def extract(self, bundle: FeatureBundle) -> NumericalFeatureVector:
        """
        Public method to extract structured feature vectors from a FeatureBundle.
        """
        dtos = self._read_dtos(bundle)
        numerical = self._extract_numerical(dtos)
        categorical = self._extract_categorical(dtos)
        embeddings = self._extract_embeddings(dtos)
        return self._produce_vector(numerical, categorical, embeddings)


class FTTransformerFeatureEmbedding(nn.Module, IFeatureEmbedding):

    def __init__(
        self,
        vector_size: int = 64,
        numerical_keys: List[str] = NUMBERS,
        categorical_keys: List[str] = CATEGORIES,
        categorical_vocabs: Optional[Dict[str, List[str]]] = None,
        text_emb_dim: int = 1536,
        vision_emb_dim: int = 512,
        audio_emb_dim: int = 128,
    ):
        super().__init__()
        self.vector_size = vector_size
        self.numerical_keys = numerical_keys
        self.categorical_keys = categorical_keys
        
        # Setup vocabulary maps
        vocabs = categorical_vocabs or VOCABS
        self.vocab_maps = {
            key: {val: idx for idx, val in enumerate(vocab)}
            for key, vocab in vocabs.items()
        }
        
        # 1. Numerical Feature Projection Parameters
        # Uses feature-specific linear project (weight/bias per feature)
        self.weights = nn.Parameter(torch.randn(len(numerical_keys), vector_size) * 0.02)
        self.biases = nn.Parameter(torch.zeros(len(numerical_keys), vector_size))
        
        # 2. Categorical Feature Embeddings
        self.embeddings = nn.ModuleDict()
        for key in self.categorical_keys:
            vocab_size = len(self.vocab_maps[key])
            self.embeddings[key] = nn.Embedding(vocab_size, vector_size)
            nn.init.normal_(self.embeddings[key].weight, std=0.02)
            
        # 3. Pre-existing Embeddings Linear Projections
        self.text_proj = nn.Linear(text_emb_dim, vector_size)
        self.vision_proj = nn.Linear(vision_emb_dim, vector_size)
        self.audio_proj = nn.Linear(audio_emb_dim, vector_size)

    def _project_numerical(self, numerical_features: Dict[str, Any]) -> torch.Tensor:
        """Projects numerical features into dense vector space."""
        first_val = next(iter(numerical_features.values())) if numerical_features else 0.0
        
        # Handle batched tensors
        if isinstance(first_val, torch.Tensor):
            batch_size = first_val.size(0)
            device = first_val.device
            feats = []
            for k in self.numerical_keys:
                val = numerical_features.get(k)
                if val is None:
                    val = torch.zeros(batch_size, device=device)
                feats.append(val.float())
            x_num = torch.stack(feats, dim=1) # (batch_size, num_num_keys)
        # Handle batched lists
        elif isinstance(first_val, list):
            batch_size = len(first_val)
            feats = []
            for k in self.numerical_keys:
                val = numerical_features.get(k, [0.0] * batch_size)
                feats.append(val)
            x_num = torch.tensor(feats, dtype=torch.float32).t()
        # Handle single instance
        else:
            lst = [float(numerical_features.get(k, 0.0)) for k in self.numerical_keys]
            x_num = torch.tensor(lst, dtype=torch.float32).unsqueeze(0) # (1, num_num_keys)
            
        # Project features individually using parallelized broadcast multiplication
        x_num = x_num.to(self.weights.device).unsqueeze(-1) # (batch_size, num_keys, 1)
        projected = (x_num * self.weights) + self.biases     # (batch_size, num_keys, vector_size)
        return projected

    def _handle_categorical(self, categorical_features: Dict[str, Any]) -> torch.Tensor:
        """Looks up class embeddings for categorical features."""
        first_val = next(iter(categorical_features.values())) if categorical_features else "UNK"
        device = next(self.embeddings.parameters()).device
        
        # Handle batched lists
        if isinstance(first_val, list):
            batch_size = len(first_val)
            indices_list = []
            for key in self.categorical_keys:
                vocab_map = self.vocab_maps[key]
                vals = categorical_features.get(key, ["UNK"] * batch_size)
                idxs = [vocab_map.get(v, 0) for v in vals]
                indices_list.append(torch.tensor(idxs, dtype=torch.long, device=device))
            x_cat = torch.stack(indices_list, dim=1)
        # Handle batched tensors
        elif isinstance(first_val, torch.Tensor):
            batch_size = first_val.size(0)
            indices_list = []
            for key in self.categorical_keys:
                val_tensor = categorical_features.get(key)
                if val_tensor is None:
                    val_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
                indices_list.append(val_tensor.long())
            x_cat = torch.stack(indices_list, dim=1)
        # Handle single instance
        else:
            lst = []
            for k in self.categorical_keys:
                val = categorical_features.get(k, "UNK")
                idx = self.vocab_maps[k].get(val, 0)
                lst.append(idx)
            x_cat = torch.tensor(lst, dtype=torch.long, device=device).unsqueeze(0)

        # Lookup embeddings
        embedded_list = []
        for i, key in enumerate(self.categorical_keys):
            feat_indices = x_cat[:, i]
            feat_emb = self.embeddings[key](feat_indices) # (batch_size, vector_size)
            embedded_list.append(feat_emb.unsqueeze(1))   # (batch_size, 1, vector_size)
            
        return torch.cat(embedded_list, dim=1)

    def _project_embeddings(
        self,
        embeddings: Dict[str, Any],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Projects pre-existing external embeddings (text, vision, audio)."""
        projected_list = []
        for name, val in embeddings.items():
            if val is None:
                continue
                
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                val = val.to(device).float()
                
            if val.dim() == 1:
                val = val.unsqueeze(0) # (1, dim)
                
            # If batched input, align embedding batch dimension
            if val.size(0) == 1 and batch_size > 1:
                val = val.expand(batch_size, -1)
                
            if name == "text_embedding":
                proj_val = self.text_proj(val)
            elif name == "vision_embedding":
                proj_val = self.vision_proj(val)
            elif name == "audio_embedding":
                proj_val = self.audio_proj(val)
            else:
                continue
                
            projected_list.append(proj_val.unsqueeze(1)) # (batch_size, 1, vector_size)
            
        if not projected_list:
            return torch.empty(batch_size, 0, self.vector_size, device=device)
            
        return torch.cat(projected_list, dim=1)

    def _learn_embeddings(
        self,
        num_tokens: torch.Tensor,
        cat_tokens: torch.Tensor,
        emb_tokens: torch.Tensor,
        feature_vector: NumericalFeatureVector
    ) -> TokenBundle:
        active_embedding_names = [
            name for name, val in feature_vector.embeddings.items() if val is not None
        ]
        num_group = Tokens(values=num_tokens, names=self.numerical_keys)
        cat_group = Tokens(values=cat_tokens, names=self.categorical_keys)
        emb_group = Tokens(values=emb_tokens, names=active_embedding_names)
        
        return TokenBundle(
            groups={
                "numerical": num_group,
                "categorical": cat_group,
                "external": emb_group
            }
        )

    def embed(self, feature_vector: NumericalFeatureVector) -> TokenBundle:
        num_tokens = self._project_numerical(feature_vector.numerical_features)
        cat_tokens = self._handle_categorical(feature_vector.categorical_features)
        
        batch_size = num_tokens.size(0)
        device = num_tokens.device
        
        emb_tokens = self._project_embeddings(feature_vector.embeddings, batch_size, device)
        
        return self._learn_embeddings(num_tokens, cat_tokens, emb_tokens, feature_vector)


class CognitiveStateEncoder(nn.Module):

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig.from_env()
        self.encoder = Encoder(self.config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.vector_size) * 0.02)

    def _learn_relationships(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _produce_latent(self, x: torch.Tensor, feature_names: List[str]) -> Latent:
        vector = x[:, 0]
        features = {}
        for i, name in enumerate(feature_names):
            features[name] = x[:, i + 1]
            
        return Latent(vector=vector, features=features)

    def encode(self, token_sequence: TokenBundle) -> Latent:
        x = token_sequence.tensor
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)
        
        encoded_seq = self._learn_relationships(x_with_cls)
        return self._produce_latent(encoded_seq, token_sequence.names)


class AppraisalNetwork(nn.Module):
    def __init__(self, vector_size: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vector_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 13),
            nn.Sigmoid() # Bounds appraisal intensities between 0.0 and 1.0
        )

    def _predict_dimensions(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent_vector)

    def _produce_dto(self, predictions: torch.Tensor) -> AppraisalDTO:
        preds = predictions[0].tolist()
        return AppraisalDTO(
            novelty=preds[0],
            threat=preds[1],
            reward=preds[2],
            goal_relevance=preds[3],
            importance=preds[4],
            urgency=preds[5],
            controllability=preds[6],
            familiarity=preds[7],
            confidence=preds[8],
            prediction_error=preds[9],
            information_gain=preds[10],
            agency=preds[11],
            social_importance=preds[12],
        )

    def predict(self, latent: Latent) -> AppraisalDTO:
        predictions = self._predict_dimensions(latent.vector)
        return self._produce_dto(predictions)


class AppraisalEngine(nn.Module, IAppraisal):

    def __init__(
        self,
        extractor: FeatureExtractor,
        embedding: IFeatureEmbedding,
        encoder: CognitiveStateEncoder,
        network: AppraisalNetwork,
    ):
        super().__init__()
        self.extractor = extractor
        self.embedding = embedding
        self.encoder = encoder
        self.network = network

    def evaluate(self, event: Any) -> AppraisalDTO:
        """
        Public method implementing IAppraisal. Evaluates a FeatureBundle through
        the cognitive appraisal pipeline.
        """
        return self._orchestrate(event)

    def _orchestrate(self, event: Any) -> AppraisalDTO:
        """Orchestrates the sequential flow of information through pipeline stages."""
        # Ensure event is a FeatureBundle
        if not isinstance(event, FeatureBundle):
            raise TypeError(f"AppraisalEngine expects FeatureBundle, got {type(event)}")
            
        feature_vector = self.extractor.extract(event)
        token_bundle = self.embedding.embed(feature_vector)
        latent = self.encoder.encode(token_bundle)
        appraisal_dto = self.network.predict(latent)
        
        return appraisal_dto

    def forward(self, bundle: FeatureBundle) -> torch.Tensor:
        feature_vector = self.extractor.extract(bundle)
        token_bundle = self.embedding.embed(feature_vector)
        latent = self.encoder.encode(token_bundle)
        predictions = self.network._predict_dimensions(latent.vector)
        return predictions
