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
# a certain transition group should be attempted first?
# this would likely be taken care of by the above since a transition would only be relevant once another had been chosen enough times?

#history will be set, such that if a transition keeps being taken, it will accumualte, if another is taken, it will be reset

#building:
#	chose node on priority
#	exapnd all
#		attempt to join, else make new
#			join based on stack
#			join based on whether type has already been attached
#		give node priority based on transition, even if node was created before with different priority

class Graph:
	def __init__(self) -> None:
		self.start_states: List[NodePattern] = []

class NodePattern:
	def __init__(self, node_parameters: NodeParameters = NodeParameters()):
		self.transitions: List[TransitionGroup] = []
		self.node_parameters: NodeParameters = node_parameters 
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
	@abstractmethod	
	def analyze(self) -> None:
		#will be implemented by auto if ever
		pass

@dataclass
class Transiton:
	next_pattern: NodePattern
	optional: bool = False
	priority: int = 0
	MAX_PRIORITY: int = 512 

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
