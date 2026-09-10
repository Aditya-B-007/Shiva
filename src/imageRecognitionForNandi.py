import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

try:
    from src.config import default_model_config, default_vision_config
except (ImportError, ModuleNotFoundError):
    from config import default_model_config, default_vision_config


class imageRecognitionEmbedder(nn.Module):
    def __init__(
        self,
        image_size: int = default_vision_config.image_size,
        patch_size: int = default_vision_config.patch_size,
        in_channels: int = default_vision_config.in_channels,
        d_model: int = default_model_config.ninp
    ):
        super().__init__()
        from src.config import VisionConfig
        if isinstance(image_size, VisionConfig):
            cfg = image_size
            image_size = cfg.image_size
            patch_size = cfg.patch_size
            in_channels = cfg.in_channels

        assert image_size % patch_size == 0, f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        assert d_model % 2 == 0, f"d_model ({d_model}) must be divisible by 2 for factorized 2D spatial coordinates"

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = in_channels * patch_size * patch_size 

        self.patch_proj = nn.Linear(self.patch_dim, d_model, bias=False) #Trainable linear projection for patch embeddings

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

    def get_image_transform(self, image_size: int = None):
        size = image_size if image_size is not None else self.image_size
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_input, image_size: int = None) -> torch.Tensor:
        size = image_size if image_size is not None else self.image_size
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif hasattr(image_input, "read"):
            img = Image.open(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        transform = self.get_image_transform(size)
        tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
        return tensor

    def recognize_image(
        self,
        model,
        tokenizer,
        image_input,
        device=None,
        max_new_tokens: int = 150,
        temperature: float = 0.5,
        top_k: int = 30,
        top_p: float = 0.85
    ) -> str:
        if device is None:
            device = next(model.parameters()).device
        image_tensor = self.preprocess_image(image_input, image_size=self.image_size).to(device)
        with torch.no_grad():
            visual_embeds = self(image_tensor)  
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None and hasattr(tokenizer, "tokenizer"):
                eos_id = tokenizer.tokenizer.token_to_id("</s>")

            output_tokens = model.generate(
                inputs_embeds=visual_embeds,
                prefix_len=visual_embeds.size(1),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_id
            )

            if isinstance(output_tokens, torch.Tensor):
                token_list = output_tokens[0].tolist()
            else:
                token_list = list(output_tokens)
            result_text = tokenizer.decode(token_list, skip_special_tokens=True)
            return result_text.strip()