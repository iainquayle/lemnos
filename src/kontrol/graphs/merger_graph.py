from torch import Tensor, Size
from torch.nn import Module, ModuleList
from structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any, Tuple, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

class Bound:
	def __init__(self, lower: int | float =1, upper: int | float =1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		if upper < 0 or lower < 0:
			exit("bound in negative")
		self.upper: int | float = upper
		self.lower: int | float = lower 
	def inside(self, i: int | float) -> bool:
		return i >= self.lower and i <= self.upper	
class TransitionGroup:
	class TransitionData(NamedTuple):
		optional: bool = False
		joining_transitions: Set[Any] = set()
		def __str__(self) -> str:
			return f"({self.optional} {self.joining_transitions})"
		def __repr__(self) -> str:
			return str(self)
	def __init__(self, transitions: Dict[Any, bool] =dict())  -> None:
		self.transitions: Dict[LayerConstraints, TransitionGroup.TransitionData] = {transition: TransitionGroup.TransitionData(optional, set()) for transition, optional in transitions.items()}
		self.joining_transitions: Set[LayerConstraints] =  set()
	def clear_joins(self):
		for transition in self.transitions:
			self.transitions[transition] = TransitionGroup.TransitionData(self.transitions[transition].optional, set())
		self.joining_transitions = set()
	def __str__(self) -> str:
		return f"TG{self.transitions} {self.joining_transitions}"
	def __repr__(self) -> str:
		return str(self)
#new node is made for each split, each new node will wrap what the splitter had prior
#deletion in this case should be the responsibility of the transition object,
#as it should delet its own node if the node is complete?
#perhaps when merging nodes, it just flattens all it can,
#then the transition object will delete all it can?
#
#transition will pull down new splits and wrap them
#there can be multiple splits
#unless all transitions always hold on to a node, no matter what, it just may be that the node is empty in the end 
#in which case deleteing nodes during the merge process is indeed the correct way to go
#
class SplitTreeNode:
	class TransitionGroupData(NamedTuple):
		transitions: Set[Any]
		next_node: Self | None 
		def __str__(self) -> str:
			return f"({self.transitions} {self.next_node})"
		def __repr__(self) -> str:
			return str(self)
	count = 0
	def __init__(self) -> None:
		self.splits: Dict[TransitionGroup, SplitTreeNode.TransitionGroupData] = dict() 
		self.id = str(SplitTreeNode.count)
		SplitTreeNode.count += 1
	def merge(self, other, merging_transition) -> Self:
		merge_stack = list(other.splits.items()) + list(self.splits.items())
		remaining_groups = dict()
		while len(merge_stack) > 0:
			group, group_data = merge_stack.pop()
			new_data = None
			if group in remaining_groups:
				new_data = SplitTreeNode.TransitionGroupData(
					remaining_groups[group].transitions | group_data.transitions,
					group_data.next_node)	
				group.joining_transitions.add(merging_transition)
				for transition in new_data.transitions:
					transition.joining_transitions.add(merging_transition)
			else:
				new_data = SplitTreeNode.TransitionGroupData(
					group_data.transitions,
					group_data.next_node)	
			if new_data.transitions == group.transitions:
				group.joining_transitions.add(merging_transition)
				for transition in new_data.transitions:
					transition.joining_transitions.add(merging_transition)
				merge_stack += list(group_data.next_node.splits.items())
		return self
	def add(self, group, transition, next_node =None):
		self.splits[group] = SplitTreeNode.TransitionGroupData({transition}, copy(next_node))	
	def __copy__(self) -> Any:
		new_node = SplitTreeNode()
		for group, (transitions, next_node) in self.splits.items():
			new_node.splits[group] = SplitTreeNode.TransitionGroupData(copy(transitions), copy(next_node))
		return new_node 
	def __str__(self) -> str:
		return f"STN{self.id} {self.splits}"	
	def __repr__(self) -> str:
		return str(self)	
#rename transition to state? or node?
class LayerConstraints:
	count = 0
	def __init__(self, 
	      shape_bounds: List[Bound] =[Bound()], 
			shape_coefficient_bounds: List[Bound] =[Bound()],
			activation_functions: List[Module] =[Identity()], 
			merge_method: MergeMethod =MergeMethod.ADD, 
			use_batch_norm: bool =True) -> None:
		self.next_state_groups: List[TransitionGroup] = [] 
		self.shape_bounds = shape_bounds 
		self.shape_coefficient_bounds = shape_coefficient_bounds
		self.activation_functions = activation_functions 
		self.merge_method = merge_method
		self.use_batch_norm = use_batch_norm 
		self.parents = dict()
		self.split_groups: Dict[TransitionGroup, Set[Self]] = dict()
		self.id = LayerConstraints.count
		LayerConstraints.count += 1
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.next_state_groups.append(copy(group))
		self.analyse_splits()
	def analyse_splits(self, visits: Set[Self] = set()) -> None:
		#TODO: it may infact be completely necessary to make a stack of the splits
		#1. stack all splits behind new splits
		#2 add them to splits?
		#3 attempt to unwrap each, 		
		# - if split can be merged, unwrap, add to list(or just straight del)
		
		#this will make it such that if a transitions gives a split to itself
		#then gives that split to another split of its own, the first node in that split wont be able to tak credit
		#keep unwrapping until unable to  
		#there will be duplicates in a recurssive stack, however if those are already in the set then it doesnt matter

		#may also be from this that visits is no longer needed????

		#other option is to make node that recognises a group from self, force the merger, 
		#this may not be a good solution
		for group in self.next_state_groups:
			for state in group.transitions:
				if self in state.parents:
					state.parents[self].add(group)
				else:
					state.parents[self] = {group} 
		self.split_groups = dict()
		for parent, parent_group_set in self.parents.items():
			for parent_group in parent_group_set:
				self.split_groups[parent_group] = {self} 
			for split_group, transitions in parent.split_groups.items():
				if split_group in self.split_groups:
					self.split_groups[split_group] = self.split_groups[split_group] | transitions
				else:
					self.split_groups[split_group] = transitions
		groups_to_delete = set()
		for split_group, transitions in self.split_groups.items():
			if set(split_group.transitions) == transitions:
				split_group.joining_transition = self
				groups_to_delete.add(split_group)
			if split_group in self.next_state_groups:
				groups_to_delete.add(split_group)
		for group in groups_to_delete:
			self.split_groups.pop(group) 
		if self not in visits:
			for group in self.next_state_groups:
				for state in group.transitions:
					state.analyse_splits(visits | {self})
		gc.collect()
	@abstractmethod
	def get_function(self, index: int, shape_tensor: List[int] | Size) -> Module:
		pass
	def get_full_str(self) -> str:
		return f"{self}: ({self.next_state_groups})"
	def __str__(self) -> str:
		return f"T{self.id}"
	def __repr__(self) -> str:
		return str(self)

#TODO: consider making this purely an injectable piece
class ConvTransition(LayerConstraints):
	def __init__(self,  next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		pass
	def get_function(self, index: int, mould_shape: List[int] | Size, shape_tensor: List[int] | Size):
		pass	
		pass