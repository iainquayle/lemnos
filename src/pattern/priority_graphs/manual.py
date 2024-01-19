from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, Size
from torch.nn import Module, ModuleList

from src.pattern.node_parameters import BaseParameters, IdentityParameters 
from src.pattern.commons import Bound, Index

from abc import ABC, abstractmethod
from typing import List, Set, Dict 
from typing_extensions import Self
from copy import copy

#TODO: items that need to be added:
# macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes

class Graph:
	def __init__(self) -> None:
		self.start_patterns: List[NodePattern] = []

class NodePattern:
	__slots__ = ["_node_parameters", "_transition_groups"]
	def __init__(self, node_parameters: BaseParameters):
		self._transition_groups: List[TransitionGroup] = []
		self._node_parameters: BaseParameters = node_parameters 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self._transition_groups.append(copy(group))
	@abstractmethod	
	def analyze(self) -> None:
		#paceholder for possible auto priority assignment
		pass

MAX_PRIORITY: int = 512 
@dataclass
class Transiton:
	next_pattern: NodePattern
	optional: bool = False
	priority: int = 0
	join_existing: bool = False

class TransitionGroup:
	__slots__ = ["_transitions", "_repetition_bounds"]
	def __init__(self, transitions: List[Transiton], repetition_bounds: Bound =Bound())  -> None:
		self._transitions: List[Transiton] = copy(transitions)
		self._repetition_bounds: Bound = repetition_bounds
	def set_transitions(self, transitions: List[Transiton]) -> None:
		pattern_set: Set[NodePattern] = set()
		for transition in transitions:
			if transition.next_pattern in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next_pattern)
		self._transitions = transitions
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)
