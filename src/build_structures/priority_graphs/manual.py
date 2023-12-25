from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, Size
from torch.nn import Module, ModuleList

#from src.model_structures.tree import Node, Tree

from src.model_structures.commons import Identity, MergeMethod
from src.build_structures.node_parameters import NodeParameters 
from src.build_structures.commons import Bound, Index

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
	def __init__(self, node_parameters: NodeParameters = NodeParameters()):
		self.transitions: List[TransitionGroup] = []
		self.node_parameters: NodeParameters = node_parameters 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
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

class TransitionGroup:
	def __init__(self, transitions: List[Transiton], repetition_bounds: Bound =Bound())  -> None:
		self.transitions: List[Transiton] = copy(transitions)
		self.repetition_bounds: Bound = repetition_bounds
	def set_transitions(self, transitions: List[Transiton]) -> None:
		pattern_set: Set[NodePattern] = set()
		for transition in transitions:
			if transition.next_pattern in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next_pattern)
		self.transitions = transitions
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)
