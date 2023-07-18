import torch
import torch.nn as nn
from abc import ABC, abstractmethod

#items to track
# children
# parent count
# shape
# build

#one possible method for graph, is use tree, where branches return to parent node, 
# reduces list of children with input, where each branches output is merged with the previouses output and fed into next
#or tree node contains eval branches, then a successor? evals are trees to, each only gets the raw input, the successor gets the merged outputs from them
#multiple outputs simple, just use a list merge and return exactly like the graph, but how to input multiple sources is trickier
#trees contain branching children, that are merged, then fed into the returning child which is the designated return

#inorder to make skip connection, create addition module
#will have to override forward method, dont want parents to concat
#orr, add special optional skip connection parent, seperate from list of parents
# only ever want one add skip connection anyways
#orr, make forward take in enum specifiying whether to concat or add

#consider seperating the input of inputs and skip input into seperate functions, then only evaluate at forward?
#with each call to one of those functions, forward inevitably called after?

#TODO: now figure out auto shape inference
#nodes could flex and change dimensions and request dimension changes from others, hoever that will add a lot of cycles
#nodes will have to chage based on parents, hoever this calls back into question the idea of the build method
#	once all parents added, have it attempt to build, dont change after that
#	just like a compiler, only check validity of build, dont attempt to flex for builder
class ModuleNode(nn.Module):
	CONCAT = 'concat'
	ADD = 'add'
	LIST = 'list'
	def __init__(self):
		super().__init__()
		self.module = None
		self.shape_out = torch.Size([]) 
		self.shape_in = torch.Size([])
		#should it really be doubly linked? or just use parent count?
		self.children = []
		self.parents = []
		self.inputs = []
		self.merge_method = ModuleNode.CONCAT
		self.merge_function = None
		self.built = False
	def add_child(self, child):
		self.children.append(child)
		child.add_parent(self)
		return self
	def add_parent(self, parent):
		self.parents.append(parent)
	def get_shape(self):
		return self.shape	
	def get_merge_function(self):
		if self.merge_method == ModuleNode.CONCAT:
			return lambda x: torch.cat(x, dim=1)
		elif self.merge_method == ModuleNode.ADD:
			return lambda x: sum(x)
		elif self.merge_method == ModuleNode.LIST:
			return lambda x: x 
		else:
			return None
	@staticmethod
	def dim_change(x, diff=1):
		return x.view([-1] + list(x.shape)[1+diff:]) if diff >= 0 else x.view([1] * -diff + list(x.shape))
	@staticmethod
	def dim_down(x, diff=1):
		ModuleNode.dim_change(x, -diff)	
	@staticmethod
	def dim_up(x, diff=1):
		ModuleNode.dim_change(x, diff)	
	@staticmethod
	def dim_squish_channel(x, diff=1):
		return x.view([1] * diff + [-1] + list(x.shape)[2:]) if diff > 0 else x
	def dim_to_input(self, x):
		return x.view(self.shape_in)
	@abstractmethod
	def module_constructor(self):
		pass
	@abstractmethod
	def dimension_inference(self):
		pass
	@abstractmethod
	def dimensionality():
		pass
	def build(self, module_constructor, x):
		self.inputs.append(x)
		if len(self.parents) >= len(self.inputs):
			if self.merge_method == ModuleNode.CONCAT:
				x = torch.cat(self.inputs, dim=1)
			elif self.merge_method == ModuleNode.ADD:
				x = sum(self.inputs)

				pass
	def reset_inputs(self):
		self.inputs = []
		for child in self.children:
			child.reset_inputs()
	def forward(self, x):
		if not self.built:
			return None
		if self.parents == [] and self.children == []:
			return self.module(x)
		self.inputs.append(self.dim_to_input(x))
		if len(self.inputs) >= len(self.parents):
			x = self.module(self.merge_function(self.inputs))
			if self.children == []:
				return x
			y = None
			for child in self.children:
				y = child(x)
			return y
		else:
			return None

class TestDenseNode(ModuleNode):
	def __init__(self, in_features=10, out_features=10):
		super().__init__()
		self.module = nn.Linear(in_features, out_features)
	def build(self, x):
		super().build(x)	


test1 = TestDenseNode(10)
test2 = TestDenseNode(10)
test3 = TestDenseNode(20, 20)
test1.add_child(test2)
test1.add_child(test3)
test2.add_child(test3)
print(test1(torch.rand(1, 10)))

class Conv2dNode(ModuleNode):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1) -> None:
		super(self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation