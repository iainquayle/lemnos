from torch import Tensor, Size
from torch.nn import Module, ModuleList
from structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Set, Dict 
from typing_extensions import Self
from copy import copy

#from input to output, shape tests must be passed
#max_delta would perhaps only be used in the cases of linear? so is it needed?
#min needed for all, 
#coeff needed for 2d, but what about filter size changes on? should these also take into account the total activation number

#needed:
# max and min size
# max coeffs (dont think w min is needed)
# tracking for how often the shape changes along a certain dimension 
#some form of tracking for jumps?
#really all decisions should be made off of the longest path to that point

#also need to accound for output shape pattern, perhaps input shape pattern aswell
#definitely input shape pattern, because output can be change in tree but input cannot as of right now

#change to shape groups?
#in each group have bounds on number of states it may transition to
#how to make functions auto size? how to track number of incoming states
#may create layer, then check whther to create function or whether to go back up and make a new branch

#the only way to create mutiple of a certain transition is via a looping transition
#thus branches must always come back to the same point, can just make a new transition for each branch

#for any one branching group, if that group is going to loop, it shall always on that same transition
#perhaps each transition graph should only have one dangling transition? may help with graph analysis

#with the graph analyzed and all transitions in accounted for, one can then traverse depth first until meeting a transitions where a parent has not been visited
#the one issue with managin gthis is the looping, how to tell when a visited transition was from a previous iteration
#this may bring into play somwthing where possible parents are culled if they are not in the current iteration, 
#but to get this to work, parents from above that are not inside the loop somehow have to eaither be tracked or ignored in the intial tally

#there can be a looping override for optional branching where the controlling parent can send a signal down that there is no need to return and the graph can continue
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

class Transition:
	count = 0
	def __init__(self, 
	      shape_bounds: List[Bound] =[Bound()], 
			shape_coefficient_bounds: List[Bound] =[Bound()],
			activation_functions: List[Module] =[Identity()], 
			merge_method: MergeMethod =MergeMethod.ADD, 
			use_batch_norm: bool =True) -> None:
		self.next_state_groups: List[Dict[Self, bool]] = [] 
		self.shape_bounds = shape_bounds 
		self.shape_coefficient_bounds = shape_coefficient_bounds
		self.activation_functions = activation_functions 
		self.merge_method = merge_method
		self.use_batch_norm = use_batch_norm 
		self.common_parents = None
		self.visits = set() 
		self.parents = set()
		self.id = Transition.count
		Transition.count += 1
	def add_next_state_group(self, next_states: Dict[Self, bool]) -> None:
		self.next_state_groups.append(next_states)
		self.analyse_visits()
	def analyse_visits(self) -> None:
		for group in self.next_state_groups:
			for state in group:
				state.parents.add(self)
				if state in self.visits or state == self:
					state.visits.add(state)
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