import os
import io
import math
import json
import base64
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from src.interfaces import IFeedbackLog, ICheckpointStore, IMultimodalSplicer, ITokenizer, MultimodalInputBatch
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import VisionEncoder, VisionBridge
from src.config import default_live_learning_config


# =============================================================================
# Data contract for a single live-learning experience
# =============================================================================

@dataclass
class LiveExperience:
    userPrompt: str
    correctedResponse: str
    reasoningThought: str
    imageBytes: Optional[bytes] = None


# =============================================================================
# Feedback Record Logger (implements IFeedbackLog)
# =============================================================================

class FeedbackRecordLogger(IFeedbackLog):
    def __init__(self, persistenceFilePath: str):
        self.persistenceFilePath = persistenceFilePath
        os.makedirs(os.path.dirname(self.persistenceFilePath), exist_ok=True)

    def recordExperienceToDisk(
        self,
        experience: LiveExperience,
        imageRelativePath: Optional[str] = None
    ) -> None:
        recordData = {
            "human_prompt": experience.userPrompt,
            "thought": experience.reasoningThought,
            "assistant_response": experience.correctedResponse
        }
        if imageRelativePath:
            recordData["image_path"] = imageRelativePath

        with open(self.persistenceFilePath, "a", encoding="utf-8") as archiveFile:
            archiveFile.write(json.dumps(recordData, ensure_ascii=False) + "\n")

    def loadReplayBufferSamples(self, maxSampleCount: int = 4) -> List[Dict[str, Any]]:
        loadedSamples = []
        if os.path.exists(self.persistenceFilePath):
            with open(self.persistenceFilePath, "r", encoding="utf-8") as archiveFile:
                for line in archiveFile:
                    trimmed = line.strip()
                    if trimmed:
                        try:
                            loadedSamples.append(json.loads(trimmed))
                        except json.JSONDecodeError:
                            continue
        if len(loadedSamples) > maxSampleCount:
            return loadedSamples[-maxSampleCount:]
        return loadedSamples


# =============================================================================
# Model Checkpoint Store (implements ICheckpointStore)
# =============================================================================

class ModelCheckpointStore(ICheckpointStore):

    def __init__(self, checkpointDirectoryPath: str):
        self.checkpointDirectoryPath = checkpointDirectoryPath
        os.makedirs(self.checkpointDirectoryPath, exist_ok=True)

    def saveCheckpointWeights(
        self,
        languageModel: nn.Module,
        visionBridge: Any,
        targetFilePath: str
    ) -> None:
        savePayload = {
            "model_state_dict": languageModel.state_dict(),
            "projector_state_dict": visionBridge.projector.state_dict()
        }
        torch.save(savePayload, targetFilePath)
        print(f"[+] Live learning weights successfully saved to: {targetFilePath}")


# =============================================================================
# Composite Persistence (Backwards-compatible helper)
# =============================================================================

class FeedbackPersistence(IFeedbackLog, ICheckpointStore):
    """Composite retaining legacy interface for callers needing unified persistence."""

    def __init__(self, persistenceFilePath: str, checkpointDirectoryPath: str):
        self.logger = FeedbackRecordLogger(persistenceFilePath)
        self.store = ModelCheckpointStore(checkpointDirectoryPath)

    def recordExperienceToDisk(self, experience: Any, imageRelativePath: Optional[str] = None) -> None:
        self.logger.recordExperienceToDisk(experience, imageRelativePath)

    def loadReplayBufferSamples(self, maxSampleCount: int = 4) -> List[Dict[str, Any]]:
        return self.logger.loadReplayBufferSamples(maxSampleCount)

    def saveCheckpointWeights(self, languageModel: nn.Module, visionBridge: Any, targetFilePath: str) -> None:
        self.store.saveCheckpointWeights(languageModel, visionBridge, targetFilePath)


# =============================================================================
# OnlineFeedbackTrainer — drives live single-step learning from UI corrections
# =============================================================================

class OnlineFeedbackTrainer:
    """
    Orchestrates online learning steps from user corrections.
    Fully decoupled with dependency-injected optimizer, loss function,
    feedback logger, and checkpoint store.
    """
    def __init__(
        self,
        languageModel: TransformerModel,
        visionBridge: VisionBridge,
        visionEncoder: VisionEncoder,
        tokenSplicer: IMultimodalSplicer,
        tokenizer: ITokenizer,
        imageTokenIdentifier: int,
        executionDevice: torch.device,
        learningRate: float = default_live_learning_config.learningRate,
        optimizer: Optional[Optimizer] = None,
        lossFunction: Optional[nn.Module] = None,
        feedbackLogger: Optional[IFeedbackLog] = None,
        checkpointStore: Optional[ICheckpointStore] = None,
        archiveStoragePath: Optional[str] = None,
        checkpointStoragePath: Optional[str] = None
    ):
        self.languageModel = languageModel
        self.visionBridge = visionBridge
        self.visionEncoder = visionEncoder
        self.tokenSplicer = tokenSplicer
        self.tokenizer = tokenizer
        self.imageTokenIdentifier = imageTokenIdentifier
        self.executionDevice = executionDevice

        resolvedArchive = archiveStoragePath or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "liveConversations.jsonl")
        )
        resolvedCheckpointDir = checkpointStoragePath or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints")
        )
        self.feedbackLogger = feedbackLogger or FeedbackRecordLogger(resolvedArchive)
        self.checkpointStore = checkpointStore or ModelCheckpointStore(resolvedCheckpointDir)
        self.targetCheckpointPath = os.path.join(resolvedCheckpointDir, "nandi_vision_final.pt")

        cfg = default_live_learning_config
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainableParameters = [
                {"params": self.languageModel.parameters(), "lr": learningRate},
                {"params": self.visionBridge.projector.parameters(), "lr": learningRate * cfg.projectorLrMultiplier}
            ]
            self.optimizer = AdamW(trainableParameters, weight_decay=cfg.weightDecay)

        self.lossFunction = lossFunction or nn.CrossEntropyLoss(ignore_index=-100)
        self.stepCounter = 0

    def maskUserPromptTokensForLossCalculation(
        self,
        completeTokenSequence: List[int],
        promptTokenCount: int
    ) -> List[int]:
        targetSequence = list(completeTokenSequence)
        limit = min(promptTokenCount, len(targetSequence))
        for tokenIndex in range(limit):
            targetSequence[tokenIndex] = -100
        return targetSequence

    def executeLiveTextLearningStep(
        self,
        userPrompt: str,
        correctedResponse: str,
        reasoningThought: str = "Clear and helpful answer."
    ) -> Dict[str, Any]:
        self.languageModel.train()
        cfg = default_live_learning_config

        promptText = f"User: {userPrompt.strip()}\nAssistant: <|thought|>\n"
        fullConversationText = f"{promptText}{reasoningThought.strip()}\n<|thought|>\n{correctedResponse.strip()}</s>"

        promptTokenIds = self.tokenizer.encode(promptText).ids
        fullTokenIds = self.tokenizer.encode(fullConversationText).ids

        if len(fullTokenIds) > cfg.maxSeqLen:
            fullTokenIds = fullTokenIds[:cfg.maxSeqLen]

        targetTokenIds = self.maskUserPromptTokensForLossCalculation(fullTokenIds, len(promptTokenIds))

        inputBatchTensor = torch.tensor([fullTokenIds[:-1]], dtype=torch.long, device=self.executionDevice)
        targetBatchTensor = torch.tensor([targetTokenIds[1:]], dtype=torch.long, device=self.executionDevice)

        self.optimizer.zero_grad()
        outputLogits = self.languageModel(src=inputBatchTensor)

        lossValue = self.lossFunction(
            outputLogits.view(-1, outputLogits.size(-1)),
            targetBatchTensor.view(-1)
        )

        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(self.languageModel.parameters(), max_norm=cfg.gradClip)
        self.optimizer.step()
        self.languageModel.eval()

        self.stepCounter += 1
        experience = LiveExperience(
            userPrompt=userPrompt,
            correctedResponse=correctedResponse,
            reasoningThought=reasoningThought
        )
        self.feedbackLogger.recordExperienceToDisk(experience)

        if self.stepCounter % cfg.checkpointEveryNSteps == 0:
            self.checkpointStore.saveCheckpointWeights(self.languageModel, self.visionBridge, self.targetCheckpointPath)

        return {
            "status": "success",
            "loss": float(lossValue.item()),
            "step": self.stepCounter
        }

    def executeLiveMultimodalLearningStep(
        self,
        rawImageBytes: bytes,
        userPrompt: str,
        correctedResponse: str,
        reasoningThought: str = "Observing visual scene and describing components accurately."
    ) -> Dict[str, Any]:
        self.languageModel.train()
        self.visionBridge.projector.train()
        cfg = default_live_learning_config

        openedImage = Image.open(io.BytesIO(rawImageBytes)).convert("RGB")
        pixelValues = self.visionEncoder.processor(images=openedImage, return_tensors="pt").pixel_values.to(self.executionDevice)

        promptText = f"User: {userPrompt.strip()}\nAssistant: <|thought|>\n"
        fullConversationText = f"{promptText}{reasoningThought.strip()}\n<|thought|>\n{correctedResponse.strip()}</s>"

        promptTokenIds = self.tokenizer.encode(promptText).ids
        fullTokenIds = self.tokenizer.encode(fullConversationText).ids

        if len(fullTokenIds) > cfg.maxSeqLen:
            fullTokenIds = fullTokenIds[:cfg.maxSeqLen]

        targetTokenIds = self.maskUserPromptTokensForLossCalculation(fullTokenIds, len(promptTokenIds))

        inputIdsTensor = torch.tensor([fullTokenIds], dtype=torch.long, device=self.executionDevice)
        labelsTensor = torch.tensor([targetTokenIds], dtype=torch.long, device=self.executionDevice)

        visualTokens = self.visionBridge(pixelValues)
        embeddingLayer = self.languageModel.get_input_embeddings()
        embeddingScale = math.sqrt(getattr(self.languageModel, "ninp", visualTokens.size(-1)))
        textEmbeddings = embeddingLayer(inputIdsTensor) * embeddingScale

        multimodalBatch = MultimodalInputBatch(
            input_ids=inputIdsTensor,
            text_embeddings=textEmbeddings,
            visual_tokens=visualTokens,
            image_token_id=self.imageTokenIdentifier,
            labels=labelsTensor
        )

        splicedBatch = self.tokenSplicer.splice(multimodalBatch)

        inputSplicedEmbeddings = splicedBatch.embeddings[:, :-1, :]
        targetSplicedLabels = splicedBatch.labels[:, 1:]

        self.optimizer.zero_grad()
        outputs = self.languageModel(inputs_embeds=inputSplicedEmbeddings)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        lossValue = self.lossFunction(
            logits.view(-1, logits.size(-1)),
            targetSplicedLabels.reshape(-1)
        )

        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.languageModel.parameters()) + list(self.visionBridge.projector.parameters()),
            max_norm=cfg.gradClip
        )
        self.optimizer.step()

        self.languageModel.eval()
        self.visionBridge.projector.eval()

        self.stepCounter += 1

        savedImageRelativePath = None
        try:
            imageDirectory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
            os.makedirs(imageDirectory, exist_ok=True)
            imageFilename = f"live_learned_{self.stepCounter:04d}.jpg"
            openedImage.save(os.path.join(imageDirectory, imageFilename))
            savedImageRelativePath = os.path.join("data", "images", imageFilename)
        except Exception:
            pass

        experience = LiveExperience(
            userPrompt=userPrompt,
            correctedResponse=correctedResponse,
            reasoningThought=reasoningThought,
            imageBytes=rawImageBytes
        )
        self.feedbackLogger.recordExperienceToDisk(experience, savedImageRelativePath)

        if self.stepCounter % cfg.checkpointEveryNSteps == 0:
            self.checkpointStore.saveCheckpointWeights(self.languageModel, self.visionBridge, self.targetCheckpointPath)

        return {
            "status": "success",
            "loss": float(lossValue.item()),
            "step": self.stepCounter
        }
