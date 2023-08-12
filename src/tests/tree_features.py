from kontrol.transitions import Transition, ConvTransition
from structures.tree import Node
from structures.commons import MergeMethod
from copy import copy
import torch.nn as nn
import torch.optim as optim
import torch

def test():
	print("Hello World!")
	mod1 = Node(function=nn.Linear(2, 2), features_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.CONCAT, split_branches=nn.ModuleList([
			Node(function=nn.Linear(2, 2), features_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE),
			Node(function=nn.Linear(2, 2), features_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE),
		]), return_branch=Node(function=nn.Linear(4, 2), features_shape=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE))

	t_in = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
	t_out = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(mod1.parameters(), lr=0.1)


	for i in range(2000):
		optimizer.zero_grad()
		output = mod1(t_in)
		print(output)
		loss = criterion(output, t_out)
		loss.backward()
		optimizer.step()
		#mod1.reset_inputs()
