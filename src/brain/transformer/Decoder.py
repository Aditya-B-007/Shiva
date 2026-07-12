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
            tensor_input = tensor_input.unsqueeze(1)
            
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
        # Build user prompt
        user_parts = [f"Perception: {context.perception}"]
        if context.emotion:
            user_parts.append(f"Current Emotion: {context.emotion}")
            
        if context.memories:
            user_parts.append("Retrieved Memories:")
            for m in context.memories:
                content = getattr(m, "raw_content", str(m))
                summary = getattr(m, "summary", "")
                user_parts.append(f"- {summary if summary else content}")
                
        if context.hypotheses:
            user_parts.append(f"Current Hypotheses: {context.hypotheses}")
            
        if context.thoughts:
            user_parts.append("Previous Thoughts:")
            for t in context.thoughts:
                user_parts.append(f"- THOUGHT: {t.thought_body}\n  CRITIQUE: {t.critique}\n  CONFIDENCE: {t.confidence}")
                
        user_parts.append("Generate the next thought, critique, and confidence:")
        user_content = "\n".join(user_parts)
        
        # System instructions
        system_content = (
            "You are the cognitive reasoning engine of a Shiva node. "
            "You must engage in active self-reflection. For each thought step, you must think and output EXACTLY in this format:\n"
            "THOUGHT: <your reasoning/thought>\n"
            "CRITIQUE: <your critique of why this might be wrong or what you missed>\n"
            "CONFIDENCE: <your self-reflected confidence between 0.0 and 1.0>\n\n"
            "If you are ready to make a final decision, append 'DECISION: <your decision>' at the end."
        )
        
        # Format using the model's native chat template
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        
        # Parse the structured self-reflection
        thought_body = ""
        critique = ""
        confidence = 0.0
        parsed_decision = None
        
        if "THOUGHT:" in raw_text:
            try:
                thought_body = raw_text.split("THOUGHT:")[-1].split("CRITIQUE:")[0].strip()
            except Exception:
                thought_body = raw_text
        else:
            thought_body = raw_text
            
        if "CRITIQUE:" in raw_text:
            try:
                critique = raw_text.split("CRITIQUE:")[-1].split("CONFIDENCE:")[0].strip()
            except Exception:
                pass
                
        if "CONFIDENCE:" in raw_text:
            try:
                conf_str = raw_text.split("CONFIDENCE:")[-1].split("DECISION:")[0].strip()
                import re
                match = re.search(r"[-+]?\d*\.\d+|\d+", conf_str)
                if match:
                    confidence = float(match.group(0))
            except Exception:
                pass
                
        if "DECISION:" in raw_text:
            try:
                parsed_decision = raw_text.split("DECISION:")[-1].strip()
            except Exception:
                pass
                
        return ThoughtDTO(
            raw_text=raw_text,
            thought_body=thought_body,
            critique=critique,
            confidence=confidence,
            parsed_decision=parsed_decision
        )
