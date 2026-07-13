import time
from typing import Optional
from ..models.ExecutionPlan import ExecutionPlan
from ..models.ToolResult import ToolResult
from ..registry.ToolRegistry import ToolRegistry
from ..tools.base import Tool

class ExecutionEngine:
    """Deterministic execution engine.
    
    Responsible only for locating, validating parameters of, and executing tools
    exactly as instructed in the ExecutionPlan.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, plan: ExecutionPlan) -> ToolResult:
        """Executes a tool according to the instructions in the ExecutionPlan.
        
        Performs validation, tool lookup, parameter type validation, and execution.
        """
        # Validate plan structure
        self._validate_plan(plan)

        # Locate tool
        tool = self.registry.get_tool(plan.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{plan.tool_name}' not found in registry.",
                timestamp=time.time()
            )

        # Validate arguments & execute
        try:
            self._validate_arguments(plan, tool)
            return tool.execute(**plan.arguments)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error on tool '{plan.tool_name}': {str(e)}",
                timestamp=time.time()
            )

    def _validate_plan(self, plan: ExecutionPlan) -> None:
        if not plan:
            raise ValueError("ExecutionPlan cannot be None")
        if not plan.tool_name:
            raise ValueError("ExecutionPlan must specify a tool_name")
        if plan.arguments is None:
            raise ValueError("ExecutionPlan arguments cannot be None")

    def _validate_arguments(self, plan: ExecutionPlan, tool: Tool) -> None:
        defs = tool.parameter_definitions
        for param_name, param_info in defs.items():
            is_required = param_info.get("required", True)
            if is_required and param_name not in plan.arguments:
                raise ValueError(f"Missing required parameter '{param_name}' for tool '{tool.name}'")

            if param_name in plan.arguments:
                val = plan.arguments[param_name]
                expected_type = param_info.get("type", "string")
                if expected_type == "integer" and not isinstance(val, int):
                    # Coerce string digits to integer
                    if isinstance(val, str) and val.isdigit():
                        plan.arguments[param_name] = int(val)
                    else:
                        raise TypeError(f"Parameter '{param_name}' must be an integer, got {type(val).__name__}")
                elif expected_type == "string" and not isinstance(val, str):
                    plan.arguments[param_name] = str(val)
