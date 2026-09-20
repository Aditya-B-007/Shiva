import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Any
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.interfaces import IVisionEncoder, IProjector, IImagePreprocessor, ITextGenerator
    from src.transformer import TextGenerator
    from src.config import default_model_config, default_vision_config, IMAGENET_MEAN, IMAGENET_STD
except (ImportError, ModuleNotFoundError):
    from interfaces import IVisionEncoder, IProjector, IImagePreprocessor, ITextGenerator
    from transformer import TextGenerator
    from config import default_model_config, default_vision_config, IMAGENET_MEAN, IMAGENET_STD

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    AutoImageProcessor, AutoModel = None, None


# =============================================================================
# Image Preprocessor (implements IImagePreprocessor)
# =============================================================================

class ImagePreprocessor(IImagePreprocessor):
    """Encapsulates image loading, PIL conversion, and torchvision normalization."""

    def __init__(self, defaultImageSize: int = default_vision_config.image_size):
        self.defaultImageSize = defaultImageSize

    def getImageTransform(self, targetImageSize: Optional[int] = None):
        size = targetImageSize if targetImageSize is not None else self.defaultImageSize
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def preprocessImage(self, rawImageInput: Any, targetImageSize: Optional[int] = None) -> torch.Tensor:
        size = targetImageSize if targetImageSize is not None else self.defaultImageSize
        if isinstance(rawImageInput, str):
            img = Image.open(rawImageInput).convert("RGB")
        elif isinstance(rawImageInput, Image.Image):
            img = rawImageInput.convert("RGB")
        elif hasattr(rawImageInput, "read"):
            img = Image.open(rawImageInput).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(rawImageInput)}")

        transform = self.getImageTransform(size)
        tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
        return tensor


# =============================================================================
# Concrete Vision Encoder (implements IVisionEncoder)
# =============================================================================

class VisionEncoder(IVisionEncoder):
    """
    Frozen visual backbone.
    Can be initialized with pre-loaded components (DIP compliant)
    or via modelName (for backwards compatibility).
    """
    def __init__(
        self,
        modelName: Optional[str] = None,
        visionModel: Optional[nn.Module] = None,
        processor: Optional[Any] = None
    ):
        super().__init__()
        if visionModel is not None and processor is not None:
            self.processor = processor
            self.vision_model = visionModel.vision_model if hasattr(visionModel, "vision_model") else visionModel
        else:
            # Fallback to direct loading for backwards compatibility
            resolvedModelName = modelName or getattr(default_vision_config, "model", "google/siglip-base-patch16-224")
            self.processor = AutoImageProcessor.from_pretrained(resolvedModelName)
            loadedModel = AutoModel.from_pretrained(resolvedModelName)
            self.vision_model = loadedModel.vision_model if hasattr(loadedModel, "vision_model") else loadedModel

        for param in self.vision_model.parameters():
            param.requires_grad = False

    @property
    def hiddenDim(self) -> int:
        return self.vision_model.config.hidden_size

    def extractFeatures(self, pixelValues: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixelValues)
            return outputs.last_hidden_state

    def forward(self, pixelValues: torch.Tensor) -> torch.Tensor:
        return self.extractFeatures(pixelValues)


# =============================================================================
# Vision Encoder Factory
# =============================================================================

class VisionEncoderFactory:
    """Factory creating VisionEncoder without constructor I/O side effects."""

    @staticmethod
    def createVisionEncoder(modelName: Optional[str] = None) -> VisionEncoder:
        resolvedModelName = modelName or getattr(default_vision_config, "model", "google/siglip-base-patch16-224")
        processor = AutoImageProcessor.from_pretrained(resolvedModelName)
        loadedModel = AutoModel.from_pretrained(resolvedModelName)
        return VisionEncoder(visionModel=loadedModel, processor=processor)


# =============================================================================
# Concrete MLP Projector (implements IProjector)
# =============================================================================

class MLPProjector(IProjector):
    def __init__(self, visual_dim: int, language_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(visual_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim)
        )

    def project(self, visualFeatures: torch.Tensor) -> torch.Tensor:
        return self.network(visualFeatures)

    def forward(self, visualFeatures: torch.Tensor) -> torch.Tensor:
        return self.project(visualFeatures)


# =============================================================================
# VisionBridge — composes encoder + projector
# =============================================================================

class VisionBridge(nn.Module):
    def __init__(self, encoder: IVisionEncoder, projector: IProjector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, pixelValues: torch.Tensor) -> torch.Tensor:
        rawFeatures = self.encoder(pixelValues)
        return self.projector(rawFeatures)


# =============================================================================
# imageRecognitionEmbedder — patch-based visual embedder
# =============================================================================

class imageRecognitionEmbedder(nn.Module):
    """
    Patch projection module with 2D factorized spatial position embeddings.
    Delegates preprocessing to ImagePreprocessor and generation to TextGenerator.
    """
    def __init__(
        self,
        image_size: int = default_vision_config.image_size,
        patch_size: int = default_vision_config.patch_size,
        in_channels: int = default_vision_config.in_channels,
        d_model: int = default_model_config.ninp,
        imagePreprocessor: Optional[IImagePreprocessor] = None
    ):
        super().__init__()
        try:
            from src.config import VisionConfig
        except ImportError:
            from config import VisionConfig

        if isinstance(image_size, VisionConfig):
            cfg = image_size
            image_size = cfg.image_size
            patch_size = cfg.patch_size
            in_channels = cfg.in_channels

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) must be divisible by 2 for factorized 2D spatial coordinates"
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.imagePreprocessor = imagePreprocessor or ImagePreprocessor(defaultImageSize=image_size)

        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Linear(self.patch_dim, d_model, bias=False)

        coord_dim = d_model // 2
        self.pos_embed_x = nn.Parameter(torch.randn(1, 1, self.grid_w, coord_dim) * 0.02)
        self.pos_embed_y = nn.Parameter(torch.randn(1, self.grid_h, 1, coord_dim) * 0.02)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        assert H == self.image_size and W == self.image_size, (
            f"Expected image resolution ({self.image_size}, {self.image_size}), got ({H}, {W})"
        )
        p = self.patch_size
        gh, gw = self.grid_h, self.grid_w

        patches = images.view(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, self.patch_dim)
        projected = self.patch_proj(patches)

        pos_x = self.pos_embed_x.expand(1, gh, gw, -1)
        pos_y = self.pos_embed_y.expand(1, gh, gw, -1)
        pos_2d = torch.cat([pos_x, pos_y], dim=-1).reshape(1, gh * gw, self.d_model)

        projected = projected + pos_2d.to(dtype=projected.dtype, device=projected.device)
        return self.norm(projected)

    def getImageTransform(self, image_size: Optional[int] = None):
        """Delegates to the injected imagePreprocessor."""
        return self.imagePreprocessor.getImageTransform(image_size)

    def preprocessImage(self, imageInput: Any, image_size: Optional[int] = None) -> torch.Tensor:
        """Delegates to the injected imagePreprocessor."""
        return self.imagePreprocessor.preprocessImage(imageInput, image_size)

    def recognizeImage(
        self,
        model,
        tokenizer,
        imageInput: Any,
        device=None,
        max_new_tokens: int = 150,
        temperature: float = 0.5,
        top_k: int = 30,
        top_p: float = 0.85
    ) -> str:
        """Convenience method for end-to-end multimodal captioning."""
        if device is None:
            device = next(model.parameters()).device
        imageTensor = self.preprocessImage(imageInput, image_size=self.image_size).to(device)
        with torch.no_grad():
            visualEmbeds = self(imageTensor)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None and hasattr(tokenizer, "tokenizer"):
                eos_id = tokenizer.tokenizer.token_to_id("</s>")

            generator = getattr(model, "_generator", None) or TextGenerator(model)
            outputTokens = generator.generateTokens(
                inputsEmbeds=visualEmbeds,
                prefixLength=visualEmbeds.size(1),
                maxNewTokens=max_new_tokens,
                temperature=temperature,
                topK=top_k,
                topP=top_p,
                eosTokenId=eos_id
            )

            if isinstance(outputTokens, torch.Tensor):
                tokenList = outputTokens[0].tolist()
            else:
                tokenList = list(outputTokens)
            resultText = tokenizer.decode(tokenList, skipSpecialTokens=True)
            return resultText.strip()