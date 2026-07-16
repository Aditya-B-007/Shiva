from __future__ import annotations

from typing import Any, Dict, List, Mapping

from src.contracts import (
    BlockCategory,
    GraphBlock,
    GraphEdge,
    OutputFormat,
    WorkflowBlock,
    WorkflowGraph,
    WorkflowRequest,
)


def compile_graph_topologically(
    blocks: Mapping[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
) -> List[str]:
    adj = {block_id: [] for block_id in blocks}
    in_degree = {block_id: 0 for block_id in blocks}

    for edge in edges:
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        if source_id in adj and target_id in adj:
            adj[source_id].append(target_id)
            in_degree[target_id] += 1

    queue = [block_id for block_id in blocks if in_degree[block_id] == 0]
    sorted_order = []

    while queue:
        current = queue.pop(0)
        sorted_order.append(current)
        for next_block in adj[current]:
            in_degree[next_block] -= 1
            if in_degree[next_block] == 0:
                queue.append(next_block)

    if len(sorted_order) < len(blocks):
        raise ValueError("Cycle detected in the workflow connections! Workflows must be Directed Acyclic Graphs (DAGs).")

    return sorted_order


def build_workflow_request(
    blocks: Mapping[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
    block_definitions: Mapping[str, Dict[str, Any]],
    query: str,
) -> WorkflowRequest:
    if not blocks:
        raise ValueError("Workflow must contain at least one block.")

    sorted_nodes = compile_graph_topologically(blocks, edges)
    has_input = False
    has_output = False
    output_format = OutputFormat.TEXT
    execution_blocks: List[WorkflowBlock] = []

    for block_id in sorted_nodes:
        block = blocks[block_id]
        definition = block_definitions[block["type"]]
        category = definition["category"]

        if category == BlockCategory.INPUT.value:
            has_input = True
        elif category == BlockCategory.OUTPUT.value:
            has_output = True
            raw_format = block["arguments"].get("output_format", OutputFormat.TEXT.value)
            try:
                output_format = OutputFormat(str(raw_format).lower())
            except ValueError:
                output_format = OutputFormat.TEXT

        execution_device = definition.get("execution_device")
        if execution_device:
            execution_blocks.append(
                WorkflowBlock(
                    device=execution_device,
                    arguments=dict(block["arguments"]),
                )
            )

    if not has_input:
        raise ValueError("Workflow must contain at least one Input block.")
    if not has_output:
        raise ValueError("Workflow must contain a Shiva Output block.")

    custom_graph = WorkflowGraph(
        blocks=[
            GraphBlock(
                id=block["id"],
                type=block["type"],
                name=block["name"],
                arguments=dict(block["arguments"]),
                condition=dict(block.get("condition", {})),
            )
            for block in blocks.values()
        ],
        edges=[
            GraphEdge(
                id=edge["id"],
                source_block_id=edge["source_id"],
                target_block_id=edge["target_id"],
            )
            for edge in edges
        ],
    )

    return WorkflowRequest(
        query=query or "Execute custom workflow.",
        blocks=execution_blocks,
        output_format=output_format,
        metadata={"custom_graph": custom_graph.to_json()},
    )
