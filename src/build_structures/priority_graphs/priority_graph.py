from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from model_structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any, Tuple, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

#it kind of seems like graph priority needs to update retro activly
#in the case of u net, the priority of the opposing side would need to always be greater, they can be stacked which solves some issues
#in the case of overlappying transitions, the lower the node, the priority needs to go up 

#when building based on priority, how will it be decided when a node cant be joined onto again?
#simply by whether it has created any children yet

#when a node is first reached, it pulls down, which sets the priority that will be passed on to subsequent nodes

#while this will give the overall build priority, for the tree, it will still need to move ahead and find which node is the merge node
#the graph should work fine, simply build in order

#building:
#	chose node on priority
#	exapnd all
#		attempt to join, else make new
#			join based on stack
#			join based on whether type has already been attached
#		give node priority based on transition, even if node was created before with different priority


class TransitionGroup:
	class TransitionData(NamedTuple):
		optional: bool = False
		#may not need this, could set up such that so long as a transition hasnt already happened from another, then it can join
		def __str__(self) -> str:
			return f"({self.optional})"
		def __repr__(self) -> str:
			return str(self)
	def __init__(self, transitions: Dict[State, bool] =dict())  -> None:
		self.transitions: Dict[State, bool] = copy(transitions)
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)

class TransitionTreeNode:
	class TransitionGroupData(NamedTuple):
		transitions: Set[State]
		next_node: Self | None 
		def __str__(self) -> str:
			return f"({self.transitions} {self.next_node})"
		def __repr__(self) -> str:
			return str(self)
	count = 0
	def __init__(self) -> None:
		self.splits: Dict[TransitionGroup, TransitionTreeNode.TransitionGroupData] = dict() 
		self.id = str(TransitionTreeNode.count)
		TransitionTreeNode.count += 1
	def merge(self, other, merging_transition) -> Self:
		#TODO
		return self	
	def add(self, group, transition, next_node =None):
		pass
	def __copy__(self) -> Any:
		new_node = TransitionTreeNode()
		for group, (transitions, next_node) in self.splits.items():
			new_node.splits[group] = TransitionTreeNode.TransitionGroupData(copy(transitions), copy(next_node))
		return new_node 
	def __str__(self) -> str:
		return f"STN{self.id} {self.splits}"	
	def __repr__(self) -> str:
		return str(self)	

class State:
	def __init__(self):
		self.transitions: List[TransitionGroup] = []
		self.parents: Dict[State, TransitionGroup] = dict()
		self.priority: int = 0
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
		self.analyze()
	def analyze(self, visited: Set[State] = set()) -> Set[Self]:#using Self typing visited results in lsp error
		self.priority = 0
		for transition_group in self.transitions:
			for transition in transition_group.transitions.keys():
				transition.parents[self] = transition_group
		if not self in visited:
			visited.add(self)
			for transition_group in self.transitions:
				for transition in transition_group.transitions.keys():
					visited = transition.analyze(visited)
		return visited
