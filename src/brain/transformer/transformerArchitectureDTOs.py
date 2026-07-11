from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

@dataclass(slots=True)
class EncoderInputDTO:
    text: Union[str, List[str]]
    max_length: int = 512

@dataclass(slots=True)
class EncoderOutputDTO:
    last_hidden_state_pt: torch.Tensor
    last_hidden_state_np: np.ndarray
    pooler_output_pt: Optional[torch.Tensor] = None
    pooler_output_np: Optional[np.ndarray] = None

@dataclass(slots=True)
class DecoderInputDTO:
    vector_input: Union[torch.Tensor, np.ndarray]

@dataclass(slots=True)
class DecoderOutputDTO:
    logits_pt: torch.Tensor
    logits_np: np.ndarray
    hidden_states_pt: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
    hidden_states_np: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None
