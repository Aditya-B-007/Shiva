from __future__ import annotations
import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger("shiva.orchestrator")

# Safe platform imports
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
    logger.warning("PyTorch not available in this environment.")

# Shiva module imports
from src.body.registry.PerceptionRegistry import PerceptionRegistry
from src.body.engine.ExecutionEngine import ExecutionEngine
from src.body.decoder.Decoder import Decoder as BodyDecoder
from src.body.perception.permissions import PermissionManager
from src.body.perception.camera import CameraDevice
from src.body.perception.microphone import MicrophoneDevice
from src.body.perception.screen import ScreenDevice
from src.body.perception.clipboard import ClipboardDevice
from src.body.perception.filesystem import FilesystemDevice
from src.body.perception.network import NetworkManager

# Brain module imports
from src.brain.memory.MemoryEngine import MemoryEngine
from src.brain.emotionalHandlerAndStore.Homeostasis import Homeostasis
from src.brain.emotionalHandlerAndStore.emotionHandlerAndOrchestrator import EmotionalOrchestrator
from src.brain.node.nodeProcessingEngine import ReasoningScheduler
from src.swarm.mothership import Mothership

# Dynamic imports for brain dynamics models which require PyTorch
try:
    from src.brain.emotionalHandlerAndStore.AppraisalEngine import (
        AppraisalEngine,
        FeatureExtractor,
        FTTransformerFeatureEmbedding,
        CognitiveStateEncoder,
        AppraisalNetwork
    )
    from src.brain.emotionalHandlerAndStore.EmotionDynamicsEngine import (
        EmotionDynamicsEngine,
        EmotionInputBuilder,
        EmotionEmbedding,
        EmotionModel,
        EmotionGenerator
    )
    from src.brain.transformer.Decoder import Decoder as TransformerDecoder
    brain_models_available = True
except ImportError as e:
    brain_models_available = False
    logger.warning(f"Cognitive deep learning models not fully loaded: {e}")

from src.orchestrator.schema import Workflow, WorkflowBlock

class ShivaOrchestrator:
    """
    Main orchestration spine of Shiva.
    Integrates perception inputs, cognitive swarms, memory, homeostasis, 
    and multi-modal decoding. Exposes a clean, direct API suitable for
    embedded wrappers (iOS/Android) and server interfaces.
    """

    def __init__(self, tts_output_dir: Optional[str] = None) -> None:
        self.permission_manager = PermissionManager()
        self.registry = PerceptionRegistry()
        self._register_default_devices()
        self.execution_engine = ExecutionEngine(self.registry)
        
        # Instantiate Body Decoder
        self.body_decoder = BodyDecoder(tts_output_dir=tts_output_dir)
        
        # Initialize Cognitive Core
        self.cognitive_initialized = False
        self.mothership: Optional[Mothership] = None
        self._init_cognitive_core()

    def _register_default_devices(self) -> None:
        """Registers default hardware perception devices."""
        try:
            self.registry.register(CameraDevice(self.permission_manager))
            self.registry.register(MicrophoneDevice(self.permission_manager))
            self.registry.register(ScreenDevice(self.permission_manager))
            self.registry.register(ClipboardDevice(self.permission_manager))
            self.registry.register(FilesystemDevice(self.permission_manager))
            self.registry.register(NetworkManager())
            logger.info("Successfully registered default perception devices.")
        except Exception as e:
            logger.error(f"Error registering default perception devices: {e}")

    def _init_cognitive_core(self) -> None:
        """Assembles cognitive swarms, memory engines, and regulatory feedback loops."""
        if not brain_models_available or torch is None:
            logger.warning("Skipping cognitive model assembly due to missing PyTorch or model dependencies.")
            return

        try:
            # 1. Initialize Memory Engine
            memory_engine = MemoryEngine()

            # 2. Initialize Homeostasis
            homeostasis = Homeostasis()

            # 3. Assemble Appraisal Engine
            extractor = FeatureExtractor()
            embedding = FTTransformerFeatureEmbedding()
            from src.brain.emotionalHandlerAndStore.AppraisalEngine import TransformerConfig
            config = TransformerConfig()
            config.vector_size = embedding.vector_size
            encoder = CognitiveStateEncoder(config=config)
            network = AppraisalNetwork(vector_size=embedding.vector_size)
            appraisal_engine = AppraisalEngine(extractor, embedding, encoder, network)

            # 4. Assemble Emotion Dynamics Engine
            input_builder = EmotionInputBuilder()
            emotion_embedding = EmotionEmbedding()
            emotion_model = EmotionModel()
            emotion_generator = EmotionGenerator()
            emotion_dynamics_engine = EmotionDynamicsEngine(
                input_builder, emotion_embedding, emotion_model, emotion_generator
            )

            # 5. Initialize Emotional Orchestrator
            emotion_handler = EmotionalOrchestrator(
                appraisal_engine=appraisal_engine,
                emotion_engine=emotion_dynamics_engine,
                homeostasis_engine=homeostasis,
                memory_engine=memory_engine
            )

            # 6. Initialize Reasoning Scheduler and Transformer Decoder
            transformer_decoder = TransformerDecoder()
            scheduler = ReasoningScheduler(transformer_decoder)

            # 7. Initialize Mothership
            self.mothership = Mothership(
                memory_engine=memory_engine,
                emotion_handler=emotion_handler,
                scheduler=scheduler,
                execution_engine=self.execution_engine
            )
            
            self.cognitive_initialized = True
            logger.info("Cognitive architecture assembled successfully.")
        except Exception as e:
            logger.error(f"Failed to assemble cognitive core: {e}", exc_info=True)

    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Returns metadata about all active hardware input devices for dynamic block UI generation."""
        devices_metadata = []
        for dev in self.registry.get_all_devices():
            devices_metadata.append({
                "name": dev.name,
                "description": dev.description,
                "parameters": dev.parameter_definitions
            })
        return devices_metadata

    def execute_workflow(self, workflow_data: dict) -> dict:
        """
        Executes a workflow JSON/dict payload and yields a formatted response DTO.
        
        Args:
            workflow_data: Dictionary representing the workflow payload.
            
        Returns:
            Dictionary payload representing the DecoderOutputDTO.
        """
        workflow = Workflow.from_dict(workflow_data)
        
        if not self.cognitive_initialized or not self.mothership:
            return {
                "format": "text",
                "text": "Cognitive core is offline (model assembly failed or dependencies missing).",
                "payload": "Cognitive core offline.",
                "metadata": {"error": True}
            }

        # 1. Compile workflow blocks into Mothership perception capture requests
        devices_to_query = []
        block_descriptions = []
        
        for block in workflow.blocks:
            devices_to_query.append({
                "device": block.device,
                "arguments": block.arguments
            })
            args_str = ", ".join([f"{k}={v}" for k, v in block.arguments.items()])
            block_descriptions.append(f"{block.device}({args_str})")

        # 2. Formulate dynamic instruction prompt combining query and workflow sequence
        compiled_query = workflow.query
        if block_descriptions:
            seq_desc = " -> ".join(block_descriptions)
            compiled_query = (
                f"User Goal: {workflow.query}\n"
                f"Workflow Execution Path: {seq_desc}\n"
                f"Please inspect the perception readings and formulate the final decision."
            )

        try:
            # 3. Solve agentically using the Mothership columns
            response_dto = self.mothership.solve_problem(
                problem=compiled_query,
                devices_to_query=devices_to_query
            )

            # 4. Decode to final output format (Text / Audio WAV payload)
            decoded_output = self.body_decoder.decode_for_output_block(
                response_dto,
                output_block=workflow.output_format
            )

            return decoded_output.to_json()
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}", exc_info=True)
            return {
                "format": "text",
                "text": f"An execution error occurred inside the swarm: {str(e)}",
                "payload": str(e),
                "metadata": {"error": True}
            }
