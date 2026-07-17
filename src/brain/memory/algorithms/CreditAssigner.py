from __future__ import annotations
from typing import List, Optional
from src.brain.memory.graph.MemoryGraph import MemoryGraph
from src.brain.memory.graph.MemoryNode import clamp_unit

class CreditAssigner:
    def __init__(self, learning_rate: float = 0.15) -> None:
        self.alpha = learning_rate

    def assign_credit(self, graph: MemoryGraph, trajectory: List[str], reward: float) -> None:
        if not trajectory or reward < 0.0:
            return

        # 1. Update individual Node strengths
        visited_nodes = set(trajectory)
        for node_id in visited_nodes:
            node = graph.get_node(node_id)
            if node is not None:
                # MC Update rule: V(S) <- V(S) + alpha * (Reward - V(S))
                current_strength = node.strength
                new_strength = current_strength + self.alpha * (reward - current_strength)
                node.strength = clamp_unit(new_strength)
                
                # Also boost activation slightly for positive reinforcement
                node.activation = clamp_unit(node.activation + 0.1 * reward)

        # 2. Update sequential transition Edge association strengths
        for i in range(len(trajectory) - 1):
            src_id = trajectory[i]
            dest_id = trajectory[i+1]
            
            # Find any connection edges between them
            for edge in graph.edges:
                if edge.source == src_id and edge.destination == dest_id:
                    current_association = edge.association_strength
                    new_association = current_association + self.alpha * (reward - current_association)
                    edge.association_strength = clamp_unit(new_association)
