import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import math

#items to track
# children
# parent count
# shape
# build

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
	def __init__(self):
		super().__init__()
		self.module = None
		self.shape_out = torch.Size([]) 
		self.shape_in = torch.Size([])
		#should it really be doubly linked? or just use parent count?
		self.children = []
		self.parents = []
		self.inputs = []
		self.merge_method = 'concat'
	def add_child(self, child):
		self.children.append(child)
		child.add_parent(self)
		return self
	def add_parent(self, parent):
		self.parents.append(parent)
	def get_shape(self):
		return self.shape	
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
	@abstractmethod
	def module_constructor(self):
		pass
	@abstractmethod
	def dimension_inference(self):
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
		if self.parents == [] and self.children == []:
			#for backwards compatibility with regular modules
			return self.module(x)
		self.inputs.append(x)
		if len(self.inputs) >= len(self.parents):
			y = []
			y = self.inputs	
			#for input in self.inputs:
				#takes care of flatten or bringing to higher dimension
			#	y.append(input.view([input.shape[0], -1, self.shape_in[2:]]))
			if self.merge_method == ModuleNode.CONCAT:
				x = torch.cat(y, dim=1)
			elif self.merge_method == ModuleNode.ADD:
				x = sum(y) 
			x = self.module(x)
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

one_d = torch.tensor(range(0, 8))
two_d = one_d.view(-1, 2)
three_d = one_d.view(2, 2, 2)
#options:
#when convert between channels, shuffle all dims up
#make for easier depthwise convolution creation, using 3d convs
#consider making convs output something with nonly 1 in first dim
#may use higher dimensional convs for all?
#if wanting 1x1 then use Dx1x1 with padding valid
def dim_test_down(x):
	return x.view([-1] + list(x.shape)[2:])
def dim_test_up(x):
	return x.view([1] + list(x.shape))
def dim_test_squish(x):
	return x.view([1, -1] + list(x.shape)[2:])


print(three_d)
print(dim_test_up(three_d))
print(dim_test_down(three_d))
print(dim_test_squish(three_d))
exit()

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