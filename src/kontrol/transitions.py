from torch import Tensor, Size
from typing import List
from abc import ABC, abstractmethod

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
class Transition:
	def __init__(self, next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		self.next_states = next_states 
		self.min_shape = min_shape 
		self.shape_coefficient_bounds = shape_coefficient_bounds
		self.activation_functions = []
	def add_next_state(self, next_state):
		self.next_states.append(next_state)
	@abstractmethod
	def get_function(index: int, shape_out: List[int] | Size, shape_tensor: List[int] | Size):
		pass
		
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