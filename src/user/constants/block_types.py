from __future__ import annotations

from enum import Enum


class BlockCategory(str, Enum):
    INPUT = "input"
    DECISION = "decision"
    OUTPUT = "output"


class InputBlockType(str, Enum):
    CAMERA = "camera"
    MICROPHONE = "microphone"
    NETWORK = "network"
    PROMPT = "prompt"


class DecisionBlockType(str, Enum):
    IF_ELSE = "if_else"
    OR = "or"
    AND = "and"


class OutputBlockType(str, Enum):
    SHIVA_OUTPUT = "shiva_output"


class OutputFormat(str, Enum):
    TEXT = "text"
    VOICE = "voice"
