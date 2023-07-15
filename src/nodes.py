import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

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

class ModuleNode(nn.Module):
	CONCAT = 'concat'
	ADD = 'add'
	def __init__(self) -> None:
		super(self).__init__()
		self.module = None
		self.shape_out = torch.Size([]) 
		self.shape_in = torch.Size([])
		self.children = []
		self.parents = []
		self.inputs = []
		self.merge_method = 'concat'
	@abstractmethod
	def add_parent(self, parent):
		pass
	def get_shape(self):
		return self.shape	
	def reset_inputs(self):
		self.inputs = []
		for child in self.children:
			child.reset_inputs()
	def forward(self, x):
		if self.parents == []:
			#for backwards compatibility with regular modules
			return self.module(x)
		self.inputs.append(x)
		if self.inputs >= len(self.parents):
			y = []	
			for input in self.inputs:
				#takes care of flatten or bringing to higher dimension
				y.append(input.view(input.shape[0].append(-1).append(self.shape_in[2:])))
			if self.merge_method == 'concat':
				x = torch.cat(y, dim=1)
			elif self.merge_method == 'add':
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

class Conv2dNode(ModuleNode):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1) -> None:
		super(self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation