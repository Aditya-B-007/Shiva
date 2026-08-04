from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from src.transferDTO import ActionInputDTO, ThoughtDTO, ThoughtParseDiagnosticsDTO, ThoughtStepDTO


class ThoughtParser:
    """
    Multi-tier resilient parsing engine for Small Language Models (SLMs ranging 0.5B - 3B).
    Exposes a single public entry point `parse(raw_text)` which orchestrates internal parsing stages.
    """

    _THOUGHT_TAG_PATTERN = re.compile(
        r"(?:\*\*)?\b(thought|critique|confidence|decision)\b(?:\*\*)?\s*:",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, raw_text: str) -> ThoughtStepDTO:
        """
        Public orchestrator method for parsing raw LLM output into a structured ThoughtStepDTO.
        Sequentially executes internal parsing stages with zero unhandled exceptions.
        """
        text = (raw_text or "").strip()
        if not text:
            return ThoughtStepDTO(
                reasoning="Empty response received",
                action="scratchpad_note",
                action_input=ActionInputDTO(raw_output=""),
                confidence=0.0,
            )

        # Stage 1: XML Thinking Blocks (<think>...</think> or unclosed <think>)
        extracted_think, cleaned_text = cls._extract_xml_think_blocks(text)

        # Stage 2 & 3: JSON Candidate Extraction (Markdown code fences or Regex)
        json_candidate = cls._extract_markdown_json(cleaned_text or text)
        if not json_candidate:
            json_candidate = cls._extract_regex_json(cleaned_text or text)

        # Stage 4: Pre-Clean JSON Candidate & Validate with Pydantic
        if json_candidate:
            cleaned_json = cls._pre_clean_json(json_candidate)
            step_dto = cls._try_parse_pydantic(cleaned_json, extracted_think)
            if step_dto is not None:
                return step_dto

        # Stage 5: Legacy Tag-Based Parsing
        legacy_dto = cls._parse_legacy_tags(cleaned_text or text, extracted_think)
        if legacy_dto is not None:
            return legacy_dto

        # Emergency Fallback (Zero crash guarantee)
        final_reasoning = extracted_think or text
        return ThoughtStepDTO(
            reasoning=final_reasoning if len(final_reasoning) <= 500 else f"{final_reasoning[:497]}...",
            action="scratchpad_note",
            action_input=ActionInputDTO(raw_output=text),
            confidence=0.3,
        )

    # --------------------------------------------------------------------------
    # PRIVATE INTERNAL STAGE METHODS
    # --------------------------------------------------------------------------

    @classmethod
    def _extract_xml_think_blocks(cls, text: str) -> Tuple[str, str]:
        extracted_think = ""
        cleaned_text = text

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            extracted_think = think_match.group(1).strip()
            cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            unclosed_match = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
            if unclosed_match:
                extracted_think = unclosed_match.group(1).strip()
                cleaned_text = ""

        return extracted_think, cleaned_text

    @classmethod
    def _extract_markdown_json(cls, text: str) -> Optional[str]:
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            return match.group(1).strip()
        return None

    @classmethod
    def _extract_regex_json(cls, text: str) -> Optional[str]:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0).strip()
        return None

    @classmethod
    def _pre_clean_json(cls, json_str: str) -> str:
        s = json_str.strip()
        s = re.sub(r",\s*([\}\]])", r"\1", s)
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    @classmethod
    def _try_parse_pydantic(cls, json_str: str, think_prefix: str = "") -> Optional[ThoughtStepDTO]:
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None
            
            reasoning = data.get("reasoning", "")
            if think_prefix and not reasoning:
                reasoning = think_prefix
            elif think_prefix and reasoning:
                reasoning = f"{think_prefix}\n{reasoning}"

            action = str(data.get("action", "scratchpad_note"))
            action_input_raw = data.get("action_input", {})
            
            if isinstance(action_input_raw, dict):
                action_input = ActionInputDTO(**action_input_raw)
            elif isinstance(action_input_raw, str):
                action_input = ActionInputDTO(raw_output=action_input_raw)
            else:
                action_input = ActionInputDTO()

            confidence_raw = data.get("confidence", 1.0)
            try:
                confidence = float(confidence_raw)
                if confidence > 1.0 and confidence <= 100.0:
                    confidence /= 100.0
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.8

            return ThoughtStepDTO(
                reasoning=str(reasoning),
                action=action,
                action_input=action_input,
                confidence=confidence,
            )
        except Exception:
            return None

    @classmethod
    def _parse_legacy_tags(cls, text: str, think_prefix: str = "") -> Optional[ThoughtStepDTO]:
        matches = list(cls._THOUGHT_TAG_PATTERN.finditer(text))
        if not matches:
            return None

        fields: Dict[str, str] = {}
        for index, match in enumerate(matches):
            key = match.group(1).lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[start:end].strip().strip("*").strip()
            if value and key not in fields:
                fields[key] = value

        thought_body = fields.get("thought") or text.strip()
        if think_prefix:
            thought_body = f"{think_prefix}\n{thought_body}"

        critique = fields.get("critique", "")
        if critique:
            reasoning = f"{thought_body}\nCritique: {critique}"
        else:
            reasoning = thought_body

        confidence = cls._parse_legacy_confidence(fields.get("confidence", ""))
        if confidence == 0.0 and "confidence" not in fields:
            confidence = 0.75

        decision = fields.get("decision")
        action = "decision" if decision else "scratchpad_note"
        action_input = ActionInputDTO(raw_output=decision) if decision else ActionInputDTO()

        return ThoughtStepDTO(
            reasoning=reasoning,
            action=action,
            action_input=action_input,
            confidence=confidence,
        )

    @classmethod
    def _parse_legacy_confidence(cls, raw_confidence: str) -> float:
        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_confidence or "")
        if not match:
            return 0.0
        value = float(match.group(0))
        if value > 1.0 and value <= 100.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))


def parse_thought_text(raw_text: str) -> ThoughtDTO:
    """
    Top-level backward compatibility wrapper around ThoughtParser.
    Converts ThoughtStepDTO to legacy ThoughtDTO.
    """
    step = ThoughtParser.parse(raw_text)
    
    parsed_decision = None
    if step.action == "decision" and step.action_input.raw_output:
        parsed_decision = step.action_input.raw_output
    elif step.action_input.code:
        parsed_decision = f"```python\n{step.action_input.code}\n```"
    elif step.action_input.command:
        parsed_decision = step.action_input.command

    return ThoughtDTO(
        raw_text=raw_text or "",
        thought_body=step.reasoning,
        critique="",
        confidence=step.confidence,
        parsed_decision=parsed_decision,
        parse_diagnostics=ThoughtParseDiagnosticsDTO(
            parse_success=step.confidence > 0.3,
            missing_fields=[] if step.confidence > 0.3 else ["reasoning"],
            warnings=[] if step.confidence > 0.3 else ["Fallback emergency parsing engaged"],
        ),
    )
