import torch
import torch.nn as nn
from core.shiva_policy import ContinuousActor, DoubleQCritic, ContinuousSACPolicy
from core.episodic_memory import EpisodicMemory
from core.latent_alignment import LatentAligner
from core.emotional_core import EmotionalCore
from core.transformer_architecture import TransformerEncoderBlock
from swarm.SwarmAlgorithmWorkspace import SwarmCoordinator  # Imported clean workspace orchestrator

class LocalVocabularyEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, text: str, device: torch.device) -> torch.Tensor:
        # Encode as raw bytes to protect against out-of-vocab token failures
        tokens = list(text.encode('utf-8', errors='ignore'))
        if not tokens:
            tokens = [0]
        tensor_tokens = torch.tensor([tokens], dtype=torch.long, device=device) # Shape: (1, T)
        return self.embedding(tensor_tokens) # Shape: (1, T, d_model)

class ShivaBackboneWrapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.num_heads = 8
        self.block = TransformerEncoderBlock(d_model=dim, num_heads=8)
        self.tokenizer_layer = LocalVocabularyEmbedding(vocab_size=256, d_model=dim)
        
    def forward_pass(self, text_directive: str, device: torch.device):
        embedded_input = self.tokenizer_layer(text_directive, device)
        return self.block.forward_pass(embedded_input)

class ShivaRuntimeManager:
    def __init__(self, d_model: int = 512, action_dim: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.sql_memory = Hippocampus(db_path="shiva_sqlite_memory.db")
        dummy_encoders = nn.ModuleDict({"text": nn.Linear(256, d_model)})
        self.aligner = LatentAligner(encoders=dummy_encoders, d_model=d_model).to(self.device)
        self.emotional_core = EmotionalCore(latent_aligner=self.aligner, hidden_dim=d_model).to(self.device)
        self.memory = EpisodicMemory(latent_dim=d_model).to(self.device)

        self.backbone = ShivaBackboneWrapper(d_model).to(self.device)
        self.actor1 = ContinuousActor(d_model, action_dim).to(self.device)
        self.actor2 = ContinuousActor(d_model, action_dim).to(self.device)
        self.critic = DoubleQCritic(d_model, action_dim).to(self.device)
        
        self.policy = ContinuousSACPolicy(
            backbone=self.backbone,
            actor1=self.actor1,
            actor2=self.actor2,
            memory=self.memory,
            critic=self.critic,
            d_model=d_model
        ).to(self.device)

        self.swarm = SwarmCoordinator(
            latent_dim=d_model,
            n_nodes=100,
            archetype_policy=self.policy,
            archetype_emotional_core=self.emotional_core
        ).to(self.device)
        
        # Link the swarm coordinator to the root policy context
        self.policy.swarm = self.swarm

    def get_cognitive_state(self) -> dict:
        mood_name, _ = self.emotional_core.current_mood()
        homeostasis = self.emotional_core._homeostasis.vector.tolist()
        return {
            "status": "online",
            "device": str(self.device),
            "latent_dim": self.d_model,
            "emotional_state": {
                "current_mood": mood_name,
                "reason": self.emotional_core.reason_for_mood_change(),
                "drives": {
                    "arousal": round(homeostasis[0], 3),
                    "energy": round(homeostasis[1], 3),
                    "safety": round(homeostasis[2], 3),
                    "engagement": round(homeostasis[3], 3)
                }
            },
            "memory_bank_size": len(self.memory._bank),
            # Integrated telemetry validation to watch global structural trends
            "swarm_telemetry": {
                "total_allocated_nodes": self.swarm.n_nodes,
                "current_diversity_loss": round(self.swarm.get_diversity_loss().item(), 4)
            }
        }

runtime_manager = ShivaRuntimeManager()
