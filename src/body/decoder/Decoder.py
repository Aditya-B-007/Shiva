import re
from typing import Any, Dict, List, Optional
from src.brain.node.nodeDTOs import NodeReasoningResultDTO
from ..models.ToolMetadata import ToolMetadata
from ..models.ExecutionPlan import ExecutionPlan

class Decoder:
    """Translates high-level swarm decisions into concrete ExecutionPlans.
    
    Serves as the semantic bridge between brain cognition and body execution.
    """

    def format_compact_prompt(
        self,
        reasoning_result: NodeReasoningResultDTO,
        available_tools: List[ToolMetadata]
    ) -> str:
        """Formats an extremely concise, token-efficient prompt for the decoder model.
        
        Saves context length by omitting deep reasoning histories and summarizing tools.
        """
        # Format tools concisely
        tools_str = ""
        for t in available_tools:
            tools_str += f"- Tool: {t.name}\n  Desc: {t.description}\n  Params: {t.parameter_definitions}\n"

        # Pull only the final thought to keep context short and clear
        last_thought = ""
        if reasoning_result.thought_history:
            lt = reasoning_result.thought_history[-1]
            last_thought = f"Last Thought: {lt.thought_body}\nCritique: {lt.critique}"

        prompt = (
            "Available Tools:\n"
            f"{tools_str}\n"
            "Cognitive Decision:\n"
            f"Decision: {reasoning_result.decision}\n"
            f"Confidence: {reasoning_result.confidence}\n"
            f"{last_thought}\n\n"
            "Task: Select the matching tool and return the call format: EXECUTE: ToolName(param=value)."
        )
        return prompt

    def decode(
        self,
        reasoning_result: NodeReasoningResultDTO,
        available_tools: List[ToolMetadata]
    ) -> Optional[ExecutionPlan]:
        """Translates NodeReasoningResultDTO into an ExecutionPlan based on available tools.
        
        Parses action structures like: "EXECUTE: LaunchApplication(application='Safari')"
        """
        decision = reasoning_result.decision
        if not decision:
            return None

        tool_names = {t.name for t in available_tools}

        # Parse EXECUTE: ToolName(param1=val1, param2=val2...)
        match = re.search(r"EXECUTE:\s*(\w+)\((.*)\)", decision)
        if not match:
            # Fallback: simple text match
            for name in tool_names:
                if name.lower() in decision.lower():
                    return ExecutionPlan(
                        tool_name=name,
                        arguments={},
                        reasoning=f"Matched tool '{name}' in decision text.",
                        decoder_confidence=reasoning_result.confidence
                    )
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        if tool_name not in tool_names:
            raise ValueError(f"Decoder attempted to call unregistered tool: {tool_name}")

        # Parse arguments
        arguments: Dict[str, Any] = {}
        for pair in re.split(r",\s*", args_str):
            if not pair:
                continue
            if "=" in pair:
                key, val = pair.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if val.isdigit():
                    arguments[key] = int(val)
                else:
                    arguments[key] = val

        return ExecutionPlan(
            tool_name=tool_name,
            arguments=arguments,
            reasoning=f"Parsed from decision: {decision}",
            decoder_confidence=reasoning_result.confidence
        )
