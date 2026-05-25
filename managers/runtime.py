import torch
import torch.nn as nn
from core.shiva_policy import ContinuousActor, DoubleQCritic, ContinuousSACPolicy
from core.episodic_memory import EpisodicMemory
from core.latent_alignment import LatentAligner
from core.emotional_core import EmotionalCore
from core.transformer_architecture import TransformerEncoderBlock

class ShivaRuntimeManager:
    def __init__(self, d_model: int = 512, action_dim: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        dummy_encoders = nn.ModuleDict({"text": nn.Linear(256, d_model)})
        self.aligner = LatentAligner(encoders=dummy_encoders, d_model=d_model).to(self.device)
        self.emotional_core = EmotionalCore(latent_aligner=self.aligner, hidden_dim=d_model).to(self.device)
        self.memory = EpisodicMemory(latent_dim=d_model).to(self.device)

        
        class ShivaBackboneWrapper(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.num_heads=8
                self.block = TransformerEncoderBlock(d_model=dim, num_heads=8)
            def forward_pass(self, x):
                return self.block(x)
                
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
            "memory_bank_size": len(self.memory._bank)
        }

runtime_manager = ShivaRuntimeManager()
