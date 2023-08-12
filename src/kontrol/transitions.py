from torch import Tensor, Size
from torch.nn import Module, ModuleList
from structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Set, Dict 
from typing_extensions import Self
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
	def __init__(self, transitions: Dict['Transition', bool] =dict(), joining_transition: 'Transition' | None =None) -> None:
		self.transitions: Dict[Transition, bool] = transitions	
		self.joining_transition: Transition | None = joining_transition 
class Transition:
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
		self.visits = dict() 
		self.parents = dict()
		self.split_groups = set()
		self.id = Transition.count
		Transition.count += 1
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.next_state_groups.append(copy(group))
		self.analyse_visits()
	#TODO: reimplement this
	def analyse_visits(self) -> None:
		#need to:
		#1.check if split groups are satisfied, if so inform group of merge, remove from splits
		# splits should be removed from child if satisfied, parent will have to be incharge of this because of how it works below
		# when checking splits, any transform that doesnt satisfy it should change the join to none
		# this will fix itself once it everything is resolved
		for group in self.split_groups:
			pass
		#2. run through groups, parents will be dict with group set? or set of transition group tuples?
		# visits should be group key dict, with set of transitions visted in past? and set of all possible transitions to visit to get to point
		# maybe only transitions that can be visited from that point
		# this will require some merger technique of dict of sets
		# this brings in question of whther to only track open splits using this method
		# then will parents even be needed? or visits? may be useful at other points though
		for group in self.next_state_groups:
			for state in group.transitions:
				state.split_groups.add(group)
				state.parents[self] = group
				if state in self.visits or state == self:
					state.parents[state] = group
				else:
					state.visits = (state.visits | self.visits)
					state.visits.add(self)
					state.analyse_visits()
	@abstractmethod
	def get_function(index: int, shape_tensor: List[int] | Size) -> Module:
		pass
	def __str__(self) -> str:
		return f"Transition {self.id}"
	def __repr__(self) -> str:
		return str(self)

#TODO: consider making this purely an injectable piece
class ConvTransition(Transition):
	def __init__(self,  next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		super().__init__(next_states, min_shape, shape_coefficient_bounds, max_concats, max_residuals)
		self.depthwise = False 
		self.kernel = (1, 1)
		self.padding = (1, 1)
		self.stride = (1, 1)
		self.dilation = (1, 1)
		self.transpose = False 
	def get_function(index: int, mould_shape: List[int] | Size, shape_tensor: List[int] | Size):
		
		pass