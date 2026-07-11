import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel
from src.brain.transformer.transformerArchitectureDTOs import EncoderInputDTO, EncoderOutputDTO

class Encoder:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def input(self, dto: EncoderInputDTO, **kwargs) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            dto.text,
            padding=True,
            truncation=True,
            max_length=dto.max_length,
            return_tensors="pt",
            **kwargs
        ).to(self.device)

    def process(self, tokenized_input: torch.Tensor | Dict[str, torch.Tensor]) -> Any:
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
