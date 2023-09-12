from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any, Tuple, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

#TODO: maybe for priorty graph analyses, walk the graph width wise using the wrapping done in the prior system
#have a set of nodes to further, each pushes into a new set.
#once a join is found, next priority number is given to the splits associated with that join if they dont hace one assigned
#eventually a join will be found, even if one branch is way over iterated, then somehow those can be cleaned
#maybe if a split has already been seen then dont explore further to make sure memory usage doesnt go crazy
#but the shortest path to the join should be the first one, so that a residual connection is the first to

#when building based on priority, how will it be decided when a node cant be joined onto again?
#simply by whether it has created any children yet

#even if this is done depth wise, it should be that the largest priority is grabbed from the parents?
#would this work for circular graphs?

#when a node is first reached, it pulls down, which sets the priority that will be passed on to subsequent nodes

#while this will give the overall build priority, for the tree, it will still need to move ahead and find which node is the merge node
#the graph should work fine, simply build in order


#how  wo we model something like U-net?


#maybe a more explicit strucuure in terms of what is happening in a graph or tree is required?

class TransitionGroup:
	class TransitionData(NamedTuple):
		optional: bool = False
		join_existing: bool = False	
		def __str__(self) -> str:
			return f"({self.optional} {self.join_existing})"
		def __repr__(self) -> str:
			return str(self)
	def __init__(self, transitions: Dict[LayerNode, bool] =dict())  -> None:
		self.transitions: Dict[LayerNode, bool] = copy(transitions)
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)
#nodes will split, when rejoining, give node the max of the nodes it is joining from?
#after a join, the priority does not need to keep going up, exactly when this is is tricky
#it will need to be able to deal with itseld and not just keep adding more to itself
#it does not need to add to priority every time
# using tree
# explore graph, only add to priority when joining a split at a node
# when joining add 1 to the max priority found in the previous nodes
# node in question is zeroed to begin
# if no joining happening at a node, then that node gets priority 0 

#it may be possible to make this less like the merge graph
# 

class TransitionTreeNode:
	class TransitionGroupData(NamedTuple):
		transitions: Set[LayerNode]
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
	pass
class LayerNode:
	def __init__(self):
		self.transitions: List[TransitionGroup] = []
		self.parents: Dict[LayerNode, TransitionGroup] = dict()
		self.priority: int = 0
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
		self.analyze()
	def analyze(self, visited: Set[LayerNode] = set()) -> Set[Self]:
	#using Self typing visited results in lsp error
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