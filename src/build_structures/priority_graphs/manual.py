from __future__ import annotations
from dataclasses import dataclass

from torch import Tensor, Size
from torch.nn import Module, ModuleList

from src.model_structures.commons import Identity, MergeMethod
from src.build_structures.module_info import ModuleInfo
from src.build_structures.commons import Bound, Index

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

#so far best option is for transitions to have weigthts, these weights are then attached to nodes that are waiting to be expanded
#tie breaker on who to expand is when they were created, queue
#each node will have the parents that have visited, if a parent has already a new node is made, otherwise it is joined
#all nodes are stacked, new nodes on top, joining nodes join the newest created that doesnt have itself in the parents


#TODO: items that need to be added:
# macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes
# a certain transition group should be attempted first?
# this would likely be taken care of by the above since a transition would only be relevant once another had been chosen enough times?

#during build, when looking at a branch, first check if it will merge up with anything, if so, take it
#if not, check whether it should be taken based on bounds and history
#history will be set, such that if a transition keeps being taken, it will accumualte, if another is taken, it will be reset


#building:
#	chose node on priority
#	exapnd all
#		attempt to join, else make new
#			join based on stack
#			join based on whether type has already been attached
#		give node priority based on transition, even if node was created before with different priority

class BuildNode:
	def __init__(self, module_info: ModuleInfo =ModuleInfo()):
		self.transitions: List[TransitionGroup] = []
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
	@abstractmethod	
	def analyze(self) -> None:
		#will be implemented by auto if ever
		pass

@dataclass
class TransitionData:
	optional: bool = False
	priority: int = 0

class TransitionGroup:
	def __init__(self, transitions: Dict[BuildNode, TransitionData] =dict(), repetition_bounds: Bound =Bound())  -> None:
		self.transitions: Dict[BuildNode, TransitionData] = copy(transitions)
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)
