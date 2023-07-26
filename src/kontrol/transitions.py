def dimension_restriction_functions(value: float):
	coeff = lambda x: x * value
	pass

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
class Transition:
	def __init__(self, dimensionality_in=1, next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		self.dimensionality_in = dimensionality_in 
		self.next_states = next_states 
		self.min_shape = min_shape 
		self.shape_coefficient_bounds = shape_coefficient_bounds
		self.activation_functions = []
		
class ConvTransition(Transition):
	def __init__(self, dimensionality_in=1, next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		super().__init__(dimensionality_in, next_states, min_shape, shape_coefficient_bounds, max_concats, max_residuals)
		self.depthwise = False 
		self.stride = (1, 1)
		self.dilation = (1, 1)
class TransposeTransition(Transition):
	def __init__(self, dimensionality_in=1, next_states=[], min_shape=[1], shape_coefficient_bounds=[1], max_concats=0, max_residuals=0) -> None:
		super().__init__(dimensionality_in, next_states, min_shape, shape_coefficient_bounds, max_concats, max_residuals)
class Conv1DTransition(Transition):
	pass
class Conv2DTransition(Transition):
	pass
class Conv3DTransition(Transition):
	pass
class Transpose1DTransition(Transition):
	pass
class Transpose2DTransition(Transition):
	pass
class Transpose3DTransition(Transition):
	pass

class ModelMetrics:
	def __init__(self) -> None:
		self.layers = []
		self.parameters = 0 
		pass