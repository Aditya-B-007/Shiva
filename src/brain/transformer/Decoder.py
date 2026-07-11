import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.brain.transformer.transformerArchitectureDTOs import DecoderInputDTO, DecoderOutputDTO
from src.brain.node.nodeDTOs import ReasoningContextDTO, ThoughtDTO

class Decoder(nn.Module):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", device: str = None):
        super().__init__()
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.projection_layer = None

    def input(self, dto: DecoderInputDTO) -> Dict[str, torch.Tensor]:
        vector_input = dto.vector_input
        if isinstance(vector_input, np.ndarray):
            tensor_input = torch.from_numpy(vector_input).float().to(self.device)
        else:
            tensor_input = vector_input.float().to(self.device)
            
        if tensor_input.ndim == 2:
            tensor_input = tensor_input.unsqueeze(1) # [batch, 1, input_dim]
            
        current_input_dim = tensor_input.shape[-1]
        target_dim = self.model.config.hidden_size
        
        if self.projection_layer is None or self.projection_layer.in_features != current_input_dim:
            self.projection_layer = nn.Linear(current_input_dim, target_dim).to(self.device)
            nn.init.normal_(self.projection_layer.weight, std=0.02)
            nn.init.zeros_(self.projection_layer.bias)
            
        inputs_embeds = self.projection_layer(tensor_input)
        return {"inputs_embeds": inputs_embeds}

    def process(self, processed_input: Dict[str, torch.Tensor], **kwargs) -> Any:
        with torch.no_grad():
            outputs = self.model(**processed_input, **kwargs)
        return outputs

    def output(self, raw_outputs: Any) -> DecoderOutputDTO:
        logits_pt = raw_outputs.logits
        logits_np = logits_pt.detach().cpu().numpy()
        
        hidden_states_pt = getattr(raw_outputs, "hidden_states", None)
        hidden_states_np = None
        
        if hidden_states_pt is not None:
            if isinstance(hidden_states_pt, tuple):
                hidden_states_np = tuple(h.detach().cpu().numpy() for h in hidden_states_pt)
            else:
                hidden_states_np = hidden_states_pt.detach().cpu().numpy()
                
        return DecoderOutputDTO(
            logits_pt=logits_pt,
            logits_np=logits_np,
            hidden_states_pt=hidden_states_pt,
            hidden_states_np=hidden_states_np
        )

    def generateDecision(self, context: ReasoningContextDTO, max_new_tokens: int = 128, **kwargs) -> ThoughtDTO:
        """
        Formats the current ReasoningContextDTO into a prompt and generates a single ThoughtDTO step.
        """
        prompt_parts = [
            "<|im_start|>system",
            "You are the cognitive reasoning engine of a Shiva node.",
            "Your goal is to evaluate the perception, memories, and emotions, and generate the next logical thought.",
            "Keep your thought concise and focused. Do not repeat previous thoughts.",
            "When you have arrived at a final decision and confidence, write: 'DECISION: <final decision> | CONFIDENCE: <0.0 to 1.0>'",
            "<|im_end|>",
            "<|im_start|>user",
            f"Perception: {context.perception}"
        ]
        
        if context.emotion:
            prompt_parts.append(f"Current Emotion: {context.emotion}")
            
        if context.memories:
            prompt_parts.append("Retrieved Memories:")
            for m in context.memories:
                content = getattr(m, "raw_content", str(m))
                summary = getattr(m, "summary", "")
                prompt_parts.append(f"- {summary if summary else content}")
                
        if context.hypotheses:
            prompt_parts.append(f"Current Hypotheses: {context.hypotheses}")
            
        if context.thoughts:
            prompt_parts.append("Previous Thoughts:")
            for t in context.thoughts:
                prompt_parts.append(f"- {t.raw_text}")
                
        prompt_parts.extend([
            "Generate the next thought:",
            "<|im_end|>",
            "<|im_start|>assistant"
        ])
        
        prompt = "\n".join(prompt_parts)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                **kwargs
            )
            
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Simple heuristic parser for closing the loop
        parsed_decision = None
        confidence = 0.0
        if "DECISION:" in raw_text:
            try:
                parts = raw_text.split("DECISION:")[-1].split("|")
                parsed_decision = parts[0].strip()
                if len(parts) > 1 and "CONFIDENCE:" in parts[1]:
                    confidence = float(parts[1].split("CONFIDENCE:")[-1].strip())
                else:
                    confidence = 1.0
            except Exception:
                pass
                
        return ThoughtDTO(raw_text=raw_text, parsed_decision=parsed_decision, confidence=confidence)
