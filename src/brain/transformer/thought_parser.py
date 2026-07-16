from __future__ import annotations

import re
from typing import Dict

from src.brain.node.nodeDTOs import ThoughtDTO, ThoughtParseDiagnosticsDTO


THOUGHT_TAG_PATTERN = re.compile(
    r"(?:\*\*)?\b(thought|critique|confidence|decision)\b(?:\*\*)?\s*:",
    re.IGNORECASE,
)


def parse_thought_text(raw_text: str) -> ThoughtDTO:
    text = raw_text or ""
    fields: Dict[str, str] = {}
    matches = list(THOUGHT_TAG_PATTERN.finditer(text))

    for index, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip().strip("*").strip()
        if value and key not in fields:
            fields[key] = value

    thought_body = fields.get("thought") or text.strip()
    critique = fields.get("critique", "")
    parsed_decision = fields.get("decision")
    confidence = _parse_confidence(fields.get("confidence", ""))
    missing_fields = [
        field_name
        for field_name in ("thought", "critique", "confidence")
        if field_name not in fields
    ]
    warnings = []
    if matches and matches[0].group(1).lower() != "thought":
        warnings.append("Structured tags were not emitted in the preferred order.")
    if fields.get("confidence") and confidence == 0.0:
        warnings.append("Confidence field was present but could not be parsed above zero.")

    return ThoughtDTO(
        raw_text=text,
        thought_body=thought_body,
        critique=critique,
        confidence=confidence,
        parsed_decision=parsed_decision,
        parse_diagnostics=ThoughtParseDiagnosticsDTO(
            parse_success=not missing_fields,
            missing_fields=missing_fields,
            warnings=warnings,
        ),
    )


def _parse_confidence(raw_confidence: str) -> float:
    match = re.search(r"[-+]?\d*\.\d+|\d+", raw_confidence or "")
    if not match:
        return 0.0
    value = float(match.group(0))
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))
