from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MCTSNode:
    prompt: str
    attack_family: str
    goal: str
    tags: List[str] = field(default_factory=list)
    parent: Optional[MCTSNode] = field(default=None, repr=False)
    children: List[MCTSNode] = field(default_factory=list, repr=False)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    operator: str = ""
    mutator_model: str = ""
    applied_operators: set = field(default_factory=set, repr=False)

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, exploration_constant: float = 1.414) -> Optional[MCTSNode]:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def add_child(self, child: MCTSNode) -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def win_rate(self) -> float:
        return self.value / self.visits if self.visits else 0.0

    def to_dict(self) -> dict:
        return {
            'name': f"[{self.attack_family}] d={self.depth}",
            'prompt': self.prompt,
            'goal': self.goal,
            'attack_family': self.attack_family,
            'tags': self.tags,
            'depth': self.depth,
            'visits': self.visits,
            'value': round(self.value, 3),
            'win_rate': round(self.win_rate, 3),
            'operator': self.operator,
            'mutator_model': self.mutator_model,
            'applied_operators': sorted(self.applied_operators),
            'children': [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional[MCTSNode] = None) -> MCTSNode:
        node = cls(
            prompt=data['prompt'],
            attack_family=data['attack_family'],
            goal=data['goal'],
            tags=data.get('tags', []),
            depth=data['depth'],
            visits=data['visits'],
            value=data['value'],
            parent=parent,
        )
        node.operator = data.get('operator', '')
        node.mutator_model = data.get('mutator_model', '')
        node.applied_operators = set(data.get('applied_operators', []))
        node.children = [cls.from_dict(c, parent=node) for c in data.get('children', [])]
        return node

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
