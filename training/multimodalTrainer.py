"""
training/multimodalTrainer.py
=============================
Shared multimodal training primitives used by both Stage 1 and Stage 2.
Avoids code duplication between stage scripts.
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.interfaces import IMultimodalSplicer, ILossEvaluator, MultimodalInputBatch
from src.imageRecognitionForNandi import VisionBridge


# =============================================================================
# Concrete loss evaluator  (implements ILossEvaluator)
# =============================================================================

class MaskedCausalLMLoss(ILossEvaluator):

    def __init__(self, ignore_index: int = -100, shift_labels: bool = True):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.shift_labels = shift_labels

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.shift_labels:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )


# =============================================================================
# Shared multimodal training step executor
# =============================================================================

class MultimodalTrainer:
    """
    Executes a single forward-pass training step for multimodal (vision + language) batches.
    Shared by Stage 1 and Stage 2; parameterised by the injected splicer and loss evaluator.
    """

    def __init__(
        self,
        slm_model: nn.Module,
        vision_bridge: VisionBridge,
        splicer: IMultimodalSplicer,
        loss_evaluator: ILossEvaluator,
        optimizer: Optimizer,
        image_token_id: int
    ):
        self.slm = slm_model
        self.bridge = vision_bridge
        self.splicer = splicer
        self.loss_evaluator = loss_evaluator
        self.optimizer = optimizer
        self.image_token_id = image_token_id

    def execute_step(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        self.bridge.projector.train()
        visual_tokens = self.bridge(pixel_values)
        embedding_layer = self.slm.get_input_embeddings()
        ninp = getattr(self.slm, "ninp", visual_tokens.size(-1))
        scale = math.sqrt(ninp)
        text_embeddings = embedding_layer(input_ids) * scale

        batch_contract = MultimodalInputBatch(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            visual_tokens=visual_tokens,
            image_token_id=self.image_token_id,
            labels=labels
        )
        spliced = self.splicer.splice(batch_contract)
        outputs = self.slm(inputs_embeds=spliced.embeddings)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = self.loss_evaluator.evaluate(logits, spliced.labels)

        return loss
