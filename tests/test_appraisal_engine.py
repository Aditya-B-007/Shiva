import sys
import os
import torch
import pytest

# Add src and subdirectories to Python path to ensure clean import resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "brain")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "brain", "emotionalHandlerAndStore")))

from brain.emotionalHandlerAndStore.emotionalContract import (
    Event, EventType, PerceptionDTO, EnvironmentDTO, EmotionDTO,
    HomeostasisDTO, IdentityDTO, MemoryDTO, FeatureBundle, AppraisalDTO
)
from brain.emotionalHandlerAndStore.AppraisalEngine import (
    FeatureExtractor, FTTransformerFeatureEmbedding, CognitiveStateEncoder,
    AppraisalNetwork, AppraisalEngine
)
from brain.transformerArchitecture import TransformerConfig
from transferDTO import Tokens, TokenBundle, Latent

def test_feature_extractor():
    # Setup a minimal FeatureBundle with missing/None sub-DTOs
    event = Event(event_type=EventType.PERCEPTION, payload=None, source="user")
    bundle = FeatureBundle(
        event=event,
        perception=PerceptionDTO(confidence=0.9),
        environment=None,
        emotion=EmotionDTO(joy=0.8, sadness=0.1),
        homeostasis=None,
        identity=None,
        memory=None
    )
    
    extractor = FeatureExtractor()
    vector = extractor.extract(bundle)
    
    # Verify that missing DTO values are defaulted properly
    assert vector.numerical_features["joy"] == 0.8
    assert vector.numerical_features["sadness"] == 0.1
    assert vector.numerical_features["fatigue"] == 0.0
    assert vector.numerical_features["env_battery_percentage"] == 100.0
    assert vector.categorical_features["event_type"] == "PERCEPTION"
    assert vector.categorical_features["event_source"] == "user"
    assert vector.categorical_features["env_charging"] == "None"
    assert vector.embeddings == {}

def test_embedding_and_encoder_and_mlp_and_engine():
    # Setup mock data containing text embedding
    event = Event(event_type=EventType.SYSTEM_INTERACTION, payload=None, source="system")
    bundle = FeatureBundle(
        event=event,
        perception=PerceptionDTO(
            confidence=0.9, 
            text_embedding=torch.randn(1536) # 1536 dim embedding
        ),
        environment=EnvironmentDTO(battery_percentage=85.0),
        emotion=EmotionDTO(joy=0.5, dominant_emotion="joy"),
        homeostasis=HomeostasisDTO(fatigue=0.2),
        identity=None,
        memory=MemoryDTO(retrieval_confidence=0.7)
    )
    
    # 1. Feature Extractor
    extractor = FeatureExtractor()
    vector = extractor.extract(bundle)
    assert "text_embedding" in vector.embeddings
    
    # 2. Embedding Layer
    vector_size = 32
    embedding_layer = FTTransformerFeatureEmbedding(
        vector_size=vector_size,
        text_emb_dim=1536
    )
    token_bundle = embedding_layer.embed(vector)
    
    # Check simplified DTO structures
    assert isinstance(token_bundle, TokenBundle)
    assert "numerical" in token_bundle.groups
    assert "categorical" in token_bundle.groups
    assert "external" in token_bundle.groups
    
    # check shape: total features = num(29) + cat(6) + active_external(1) = 36
    assert token_bundle.tensor.size() == (1, 36, vector_size)
    assert len(token_bundle.names) == 36
    
    # 3. Cognitive State Encoder
    config = TransformerConfig(
        vector_size=vector_size,
        num_layers=2,
        num_heads=4,
        feed_forward_dimension=64
    )
    encoder = CognitiveStateEncoder(config)
    latent = encoder.encode(token_bundle)
    
    assert isinstance(latent, Latent)
    assert latent.vector.size() == (1, vector_size)
    
    # Check that individual features can be inspected in the DTO for debugging
    assert "joy" in latent.features
    assert "event_type" in latent.features
    assert "text_embedding" in latent.features
    assert latent.features["joy"].size() == (1, vector_size)
    
    # 4. Appraisal MLP Network
    network = AppraisalNetwork(vector_size=vector_size, hidden_dim=64)
    appraisal = network.predict(latent)
    
    assert isinstance(appraisal, AppraisalDTO)
    assert 0.0 <= appraisal.novelty <= 1.0
    
    # 5. Appraisal Engine Orchestrator
    engine = AppraisalEngine(
        extractor=extractor,
        embedding=embedding_layer,
        encoder=encoder,
        network=network
    )
    
    appraisal_dto = engine.evaluate(bundle)
    assert isinstance(appraisal_dto, AppraisalDTO)
    assert 0.0 <= appraisal_dto.novelty <= 1.0
    
    # 6. End-to-end Backpropagation Test
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)
    preds = engine(bundle)
    assert preds.size() == (1, 13)
    
    loss = preds.sum()
    optimizer.zero_grad()
    loss.backward()
    
    # Ensure gradients flow to all parts of the model parameters
    assert embedding_layer.weights.grad is not None
    assert encoder.cls_token.grad is not None
    assert next(network.parameters()).grad is not None
