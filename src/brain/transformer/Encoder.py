import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
from src.transferDTO import EncoderInputDTO, EncoderOutputDTO

class Encoder:
    def __init__(self, model_name: str = "google-bert/bert-large-uncased", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self._model_load_error: Optional[Exception] = None

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        if self._model_load_error is not None:
            raise RuntimeError(f"Encoder model is unavailable: {self._model_load_error}") from self._model_load_error

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            self.model.eval()
        except Exception as exc:
            self._model_load_error = exc
            raise RuntimeError(f"Encoder model is unavailable: {exc}") from exc

    def input(self, dto: EncoderInputDTO, **kwargs) -> Dict[str, torch.Tensor]:
        self._ensure_model_loaded()
        return self.tokenizer(
            dto.text,
            padding=True,
            truncation=True,
            max_length=dto.max_length,
            return_tensors="pt",
            **kwargs
        ).to(self.device)

    def process(self, tokenized_input: torch.Tensor | Dict[str, torch.Tensor]) -> Any:
        self._ensure_model_loaded()
        with torch.no_grad():
            if isinstance(tokenized_input, torch.Tensor):
                outputs = self.model(inputs_embeds=tokenized_input)
            else:
                outputs = self.model(**tokenized_input)
        return outputs

    def Output(self, raw_outputs: Any) -> EncoderOutputDTO:
        last_hidden_state_pt = raw_outputs.last_hidden_state
        pooler_output_pt = getattr(raw_outputs, "pooler_output", None)
        
        last_hidden_state_np = last_hidden_state_pt.detach().cpu().numpy()
        
        pooler_output_np = None
        if pooler_output_pt is not None:
            pooler_output_np = pooler_output_pt.detach().cpu().numpy()
            
        return EncoderOutputDTO(
            last_hidden_state_pt=last_hidden_state_pt,
            last_hidden_state_np=last_hidden_state_np,
            pooler_output_pt=pooler_output_pt,
            pooler_output_np=pooler_output_np
        )
