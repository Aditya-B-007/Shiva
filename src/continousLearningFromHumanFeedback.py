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
from torch.optim import AdamW

from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import VisionEncoder, VisionBridge
from src.dataIngestionPipeline import TokenSplicer, MultimodalInputBatch


@dataclass
class LiveExperience:
    userPrompt: str
    correctedResponse: str
    reasoningThought: str
    imageBytes: Optional[bytes] = None


class ExperienceArchive:
    def __init__(self, persistenceFilePath: str, checkpointDirectoryPath: str):
        self.persistenceFilePath = persistenceFilePath
        self.checkpointDirectoryPath = checkpointDirectoryPath
        os.makedirs(os.path.dirname(self.persistenceFilePath), exist_ok=True)
        os.makedirs(self.checkpointDirectoryPath, exist_ok=True)

    def recordExperienceToDisk(self, experience: LiveExperience, imageRelativePath: Optional[str] = None) -> None:
        recordData = {
            "human_prompt": experience.userPrompt,
            "thought": experience.reasoningThought,
            "assistant_response": experience.correctedResponse
        }
        if imageRelativePath:
            recordData["image_path"] = imageRelativePath

        with open(self.persistenceFilePath, "a", encoding="utf-8") as archiveFile:
            archiveFile.write(json.dumps(recordData, ensure_ascii=False) + "\n")

    def saveCheckpointWeights(self, languageModel: TransformerModel, visionBridge: VisionBridge, targetFilePath: str) -> None:
        savePayload = {
            "model_state_dict": languageModel.state_dict(),
            "projector_state_dict": visionBridge.projector.state_dict()
        }
        torch.save(savePayload, targetFilePath)
        print(f"[+] Live learning weights successfully saved to: {targetFilePath}")

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


class LiveLearner:
    def __init__(
        self,
        languageModel: TransformerModel,
        visionBridge: VisionBridge,
        visionEncoder: VisionEncoder,
        tokenSplicer: TokenSplicer,
        tokenizer: TokenizerNandi,
        imageTokenIdentifier: int,
        executionDevice: torch.device,
        learningRate: float = 1e-5,
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
        self.archive = ExperienceArchive(resolvedArchive, resolvedCheckpointDir)
        self.targetCheckpointPath = os.path.join(resolvedCheckpointDir, "nandi_vision_final.pt")

        trainableParameters = [
            {"params": self.languageModel.parameters(), "lr": learningRate},
            {"params": self.visionBridge.projector.parameters(), "lr": learningRate * 2.0}
        ]
        self.optimizer = AdamW(trainableParameters, weight_decay=0.01)
        self.lossFunction = nn.CrossEntropyLoss(ignore_index=-100)
        self.stepCounter = 0

    def maskUserPromptTokensForLossCalculation(self, completeTokenSequence: List[int], promptTokenCount: int) -> List[int]:
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

        promptText = f"User: {userPrompt.strip()}\nAssistant: <|thought|>\n"
        fullConversationText = f"{promptText}{reasoningThought.strip()}\n<|thought|>\n{correctedResponse.strip()}</s>"

        promptTokenIds = self.tokenizer.encode(promptText).ids
        fullTokenIds = self.tokenizer.encode(fullConversationText).ids

        if len(fullTokenIds) > 512:
            fullTokenIds = fullTokenIds[:512]

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
        torch.nn.utils.clip_grad_norm_(self.languageModel.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.languageModel.eval()

        self.stepCounter += 1
        experience = LiveExperience(
            userPrompt=userPrompt,
            correctedResponse=correctedResponse,
            reasoningThought=reasoningThought
        )
        self.archive.recordExperienceToDisk(experience)

        if self.stepCounter % 3 == 0:
            self.archive.saveCheckpointWeights(self.languageModel, self.visionBridge, self.targetCheckpointPath)

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

        openedImage = Image.open(io.BytesIO(rawImageBytes)).convert("RGB")
        pixelValues = self.visionEncoder.processor(images=openedImage, return_tensors="pt").pixel_values.to(self.executionDevice)

        promptText = f"User: {userPrompt.strip()}\nAssistant: <|thought|>\n"
        fullConversationText = f"{promptText}{reasoningThought.strip()}\n<|thought|>\n{correctedResponse.strip()}</s>"

        promptTokenIds = self.tokenizer.encode(promptText).ids
        fullTokenIds = self.tokenizer.encode(fullConversationText).ids

        if len(fullTokenIds) > 512:
            fullTokenIds = fullTokenIds[:512]

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

        splicedEmbeddings = splicedBatch.embeddings
        splicedLabels = splicedBatch.labels

        inputSplicedEmbeddings = splicedEmbeddings[:, :-1, :]
        targetSplicedLabels = splicedLabels[:, 1:]

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
            max_norm=1.0
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
        self.archive.recordExperienceToDisk(experience, savedImageRelativePath)

        if self.stepCounter % 3 == 0:
            self.archive.saveCheckpointWeights(self.languageModel, self.visionBridge, self.targetCheckpointPath)

        return {
            "status": "success",
            "loss": float(lossValue.item()),
            "step": self.stepCounter
        }
