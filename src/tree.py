import torch
import torch.nn as nn


identity = lambda x: x
CONCAT = 'concat'
ADD = 'add'
LIST = 'list'
def get_merge_function(merge_method):
	if merge_method == CONCAT:
		return lambda x: torch.cat(x, dim=1)
	elif merge_method == ADD:
		return lambda x: sum(x)
	elif merge_method == LIST:
		return identity 
	else:
		return None

class Tree(nn.Module):
	pass

class Node(nn.Module):
	def __init__(self, merge_method=CONCAT, function=identity, shape=torch.Size([0]), sub_branches=[identity], return_branch=identity):
		super().__init__()
		self.function = function 
		self.shape = shape 
		self.sub_branches = sub_branches 
		self.return_branch = return_branch 
		self.merge_function = get_merge_function(merge_method)
	@staticmethod
	def change_shape(shape, diff=1):
		return [-1] + list(shape)[1+diff:] if diff >= 0 else [1] * -diff + list(shape)
	@staticmethod
	def squish_shape(shape, diff=1):
		return [1] * diff + [-1] + list(shape)[2:] 
	def mould_shape(self, x):
		return x.view(self.shape)
	def forward(self, x):
		#maybe change to
		# mould
		# function
		# branches
		# return
		#or (likely somehing like this)
		#though mould first may be the way, or else nothing will change to whats needed before being inputed?
		#aslong as it is before the return branch it should be fine 
		# branches
		# mould
		# function
		# return
		return self.return_branch(
			self.function(
			self.mould_shape(
			self.merge_function(
			list(map(lambda module: module(x), self.sub_branches))))))

print("start")			
test1 = Node(function=nn.Linear(5, 5), merge_method=ADD, shape=torch.Size([5]))
print(test1(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])))