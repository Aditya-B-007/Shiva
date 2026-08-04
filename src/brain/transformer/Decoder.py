import os
import sys
import logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.transferDTO import DecoderInputDTO, DecoderOutputDTO, ReasoningContextDTO, ThoughtDTO
from src.brain.transformer.thought_parser import parse_thought_text

class Decoder(nn.Module):
    def __init__(self, model_name: str = None, device: str = None):
        super().__init__()
        resolved_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        
        if getattr(sys, 'frozen', False):
            project_root = os.path.dirname(sys.executable)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        local_cache_dir = os.path.join(project_root, "models", "shiva-decoder")
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        os.makedirs(local_cache_dir, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name,
                cache_dir=local_cache_dir,
                local_files_only=True
            )
        except Exception:
            print("[Shiva Engine] Initializing cognitive tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name,
                cache_dir=local_cache_dir,
                local_files_only=False
            )
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                cache_dir=local_cache_dir,
                local_files_only=True
            ).to(self.device)
        except Exception:
            print("[Shiva Engine] Downloading cognitive decoder weights (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                cache_dir=local_cache_dir,
                local_files_only=False
            ).to(self.device)
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

    def execute_sandbox_script(self, code: str, workspace_dir: Optional[str] = None) -> str:
        import io
        import contextlib
        import sys
        import os
        import math
        import json
        import re
        import pathlib
        import shutil
        import glob
        import random
        import datetime
        import time
        import collections
        import itertools
        import functools
        import hashlib
        import csv
        import socket
        import urllib
        import subprocess

        sandbox_globals = {
            "os": os,
            "sys": sys,
            "math": math,
            "json": json,
            "re": re,
            "pathlib": pathlib,
            "shutil": shutil,
            "glob": glob,
            "random": random,
            "datetime": datetime,
            "time": time,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "hashlib": hashlib,
            "csv": csv,
            "socket": socket,
            "urllib": urllib,
            "subprocess": subprocess,
            "WORKSPACE_DIR": workspace_dir,
        }

        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                # Execute the code block
                exec(code, sandbox_globals)
            output = stdout_buf.getvalue()
            return output if output.strip() else "Script executed successfully with no stdout."
        except Exception as e:
            return f"Sandbox execution error: {type(e).__name__}: {str(e)}"

    def generateDecision(self, context: ReasoningContextDTO, max_new_tokens: int = 32768, workspace_dir: Optional[str] = None, **kwargs) -> ThoughtDTO:
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
                
        if context.context:
            user_parts.append("Runtime Guidance:")
            for key, value in context.context.items():
                user_parts.append(f"- {key}: {value}")

        user_parts.append("Generate the next thought, critique, and confidence:")
        user_content = "\n".join(user_parts)
        
        # System instructions with JSON Schema guidance for SLMs
        system_content = (
            "You are the cognitive reasoning engine of a Shiva node. "
            "You must engage in active self-reflection. You may output your thought step as a JSON object matching this schema:\n"
            "{\n"
            '  "reasoning": "<your step-by-step reflection and critique>",\n'
            '  "action": "scratchpad_note | execute_code | decision",\n'
            '  "action_input": {"code": "<optional python block>", "raw_output": "<optional decision>"},\n'
            '  "confidence": 0.85\n'
            "}\n\n"
            "Alternatively, format as:\n"
            "THOUGHT: <your reasoning/thought>\n"
            "CRITIQUE: <your critique>\n"
            "CONFIDENCE: <0.0 to 1.0>\n"
            "DECISION: <your decision or ```python script ```>"
        )
        
        # Format using the model's native chat template
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        
        latent_action = kwargs.pop("latent_action", None)

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        }
        generation_kwargs.update(kwargs)

        if latent_action is not None:
            # 1. Project latent action vector to model hidden size
            if isinstance(latent_action, np.ndarray):
                latent_action_t = torch.from_numpy(latent_action).float().to(self.device)
            else:
                latent_action_t = latent_action.float().to(self.device)
            
            if latent_action_t.ndim == 1:
                latent_action_t = latent_action_t.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
            elif latent_action_t.ndim == 2:
                latent_action_t = latent_action_t.unsqueeze(1)  # (batch, 1, action_dim)
                
            target_dim = self.model.config.hidden_size
            if self.projection_layer is None or self.projection_layer.in_features != latent_action_t.shape[-1]:
                self.projection_layer = nn.Linear(latent_action_t.shape[-1], target_dim).to(self.device)
                nn.init.normal_(self.projection_layer.weight, std=0.02)
                nn.init.zeros_(self.projection_layer.bias)
                
            latent_embed = self.projection_layer(latent_action_t)  # (1, 1, hidden_size)
            
            # 2. Get standard prompt token embeddings
            token_embeds = self.model.get_input_embeddings()(inputs["input_ids"])  # (1, seq_len, hidden_size)
            
            # 3. Concatenate them
            inputs_embeds = torch.cat([latent_embed, token_embeds], dim=1)
            
            # 4. Construct matching attention mask
            ones = torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long, device=self.device)
            attention_mask = torch.cat([ones, inputs["attention_mask"]], dim=1)
            
            generation_kwargs["inputs_embeds"] = inputs_embeds
            generation_kwargs["attention_mask"] = attention_mask
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            generated_tokens = outputs[0]
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]

        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        thought_dto = parse_thought_text(raw_text)
        decision_body = thought_dto.parsed_decision or thought_dto.thought_body
        if decision_body and "```python" in decision_body:
            try:
                start_idx = decision_body.find("```python") + len("```python")
                end_idx = decision_body.find("```", start_idx)
                if end_idx != -1:
                    code_block = decision_body[start_idx:end_idx].strip()
                    sandbox_result = self.execute_sandbox_script(code_block, workspace_dir)
                    enriched_decision = (
                        f"{decision_body}\n\n"
                        f"--- SANDBOX EXECUTION OUTPUT ---\n"
                        f"{sandbox_result}\n"
                        f"--------------------------------"
                    )
                    if thought_dto.parsed_decision:
                        thought_dto.parsed_decision = enriched_decision
                    else:
                        thought_dto.thought_body = enriched_decision
            except Exception as e:
                logger.error(f"Error handling sandbox execution in generateDecision: {e}")

        return thought_dto
