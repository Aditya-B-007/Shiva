"""
src/interfaces.py
=================
Single source of truth for every abstract interface and shared data contract.

Rule: all business logic depends on these abstractions.
      concrete implementations live in their own modules and import from here.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# =============================================================================
# Shared Data Transfer Objects
# (owned here so both the pipeline and the trainer speak the same contract)
# =============================================================================

@dataclass(frozen=True)
class MultimodalInputBatch:
    input_ids: torch.Tensor
    text_embeddings: torch.Tensor
    visual_tokens: torch.Tensor
    image_token_id: int
    labels: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SplicedMultimodalOutput:
    embeddings: torch.Tensor
    labels: Optional[torch.Tensor] = None


# =============================================================================
# Tokenizer Interfaces
# =============================================================================

class ITokenizer(ABC):
    """Minimal read-only contract. Inference code depends on this only."""

    @abstractmethod
    def encode(self, text: str):
        """Encode text and return an encoding object with an `.ids` attribute."""
        ...

    @abstractmethod
    def decode(self, tokenIds, skipSpecialTokens: bool = False) -> str:
        """Decode a list of token IDs back to text."""
        ...

    @abstractmethod
    def getVocabSize(self) -> int:
        """Return the current vocabulary size."""
        ...


class ITrainableTokenizer(ITokenizer):
    """Extended contract for tokenizer training pipelines."""

    @abstractmethod
    def train(self, corpusPath: str) -> None:
        """Train (or retrain) the tokenizer on the given corpus file."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load a pre-trained tokenizer from disk."""
        ...

    @abstractmethod
    def addSpecialTokens(self, tokens) -> int:
        """Add special tokens and return the number added."""
        ...


# =============================================================================
# Inference & Generation Interfaces
# =============================================================================

class ITextGenerator(ABC):
    """Contract for autoregressive token sequence generation."""

    @abstractmethod
    def generateTokens(
        self,
        tokenIndices: Optional[torch.Tensor] = None,
        inputsEmbeds: Optional[torch.Tensor] = None,
        prefixLength: int = 0,
        maxNewTokens: int = 250,
        temperature: float = 0.5,
        topK: int = 30,
        topP: float = 0.85,
        repetitionPenalty: float = 1.2,
        eosTokenId: Optional[int] = None
    ) -> torch.Tensor:
        """Generate token sequence using sampling and autoregressive decoding."""
        ...


# =============================================================================
# Vision & Image Preprocessing Interfaces
# =============================================================================

class IImagePreprocessor(ABC):
    """Contract for preparing raw image inputs into normalized PyTorch tensors."""

    @abstractmethod
    def getImageTransform(self, targetImageSize: Optional[int] = None):
        """Construct and return the torchvision image transform pipeline."""
        ...

    @abstractmethod
    def preprocessImage(self, rawImageInput: Any, targetImageSize: Optional[int] = None) -> torch.Tensor:
        """Convert PIL, file path, or file stream to a normalized 4D image batch tensor."""
        ...


class IVisionEncoder(ABC, nn.Module):
    """Contract for a frozen visual backbone that extracts patch features."""

    @property
    @abstractmethod
    def hiddenDim(self) -> int:
        """Dimensionality of the feature vectors this encoder produces."""
        ...

    @abstractmethod
    def extractFeatures(self, pixelValues: torch.Tensor) -> torch.Tensor:
        """Return last-hidden-state features for a batch of pixel tensors."""
        ...


class IProjector(ABC, nn.Module):
    """Contract for projecting visual features into the language model's space."""

    @abstractmethod
    def project(self, visualFeatures: torch.Tensor) -> torch.Tensor:
        """Map visual feature vectors to language-model embedding space."""
        ...


# =============================================================================
# Data Pipeline Interfaces
# =============================================================================

class IMultimodalSplicer(ABC):
    """Contract for fusing visual tokens into a text embedding sequence."""

    @abstractmethod
    def splice(self, batch: MultimodalInputBatch) -> SplicedMultimodalOutput:
        """Replace <image> placeholder embeddings with visual token embeddings."""
        ...


# =============================================================================
# Training Interfaces
# =============================================================================

class ILossEvaluator(ABC):
    """Contract for computing a scalar training loss from logits and labels."""

    @abstractmethod
    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute and return the loss tensor."""
        ...


class ITrainingStageConfigurator(ABC):
    """Contract for configuring which parameters are trainable at each stage."""

    @abstractmethod
    def configure(self, bridge: Any, slm: nn.Module, lr: float):
        """
        Freeze/unfreeze parameters and return a configured optimizer.

        Args:
            bridge: the VisionBridge (encoder + projector composite).
            slm:    the language model nn.Module.
            lr:     base learning rate.

        Returns:
            A torch.optim.Optimizer ready for use.
        """
        ...


# =============================================================================
# Persistence Interfaces
# =============================================================================

class ICheckpointStore(ABC):
    """Contract for saving model weights to durable storage."""

    @abstractmethod
    def saveCheckpointWeights(
        self,
        languageModel: nn.Module,
        visionBridge: Any,
        targetFilePath: str
    ) -> None:
        """Persist model and projector state dicts to the given path."""
        ...


class IFeedbackLog(ABC):
    """Contract for recording and replaying human-feedback training examples."""

    @abstractmethod
    def recordExperienceToDisk(
        self,
        experience: Any,
        imageRelativePath: Optional[str] = None
    ) -> None:
        """Append a feedback record to the persistent log."""
        ...

    @abstractmethod
    def loadReplayBufferSamples(
        self,
        maxSampleCount: int = 4
    ) -> List[Dict[str, Any]]:
        """Return up to maxSampleCount recent feedback records."""
        ...
