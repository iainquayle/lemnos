import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

#items to track
# children
# parent count
# shape
# build

class ModuleNode(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.parent_count = 0
		self.module = None
		self.children = []
		self.built = False
		self.shape = None
	@abstractmethod
	def shape(self):
		pass
	@abstractmethod
	def build(self):
		pass

class Conv2dNode(ModuleNode):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1) -> None:
		super(Conv2dNode, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation