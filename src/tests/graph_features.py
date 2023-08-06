from kontrol.transitions import Transition, ConvTransition
from structures.graph import Node
from structures.commons import MergeMethod
from copy import copy
import torch.nn as nn
import torch.optim as optim
import torch

print("Hello World!")
mod1 = Node(function=nn.Linear(2, 2), shape_in=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
mod2 = Node(function=nn.Linear(2, 2), shape_in=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
mod3 = Node(function=nn.Linear(2, 2), shape_in=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.SINGLE)
mod4 = Node(function=nn.Linear(4, 2), shape_in=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.CONCAT)
mod5 = Node(function=nn.Linear(2, 2), shape_in=[2], activation=nn.Sigmoid(), merge_method=MergeMethod.ADD)

mod1.node_children = nn.ModuleList([mod2, mod3, mod5]) 
mod2.node_children = nn.ModuleList([mod4]) 
mod2.node_parents = nn.ModuleList([mod1]) 
mod3.node_children = nn.ModuleList([mod4]) 
mod3.node_parents = nn.ModuleList([mod1]) 
mod4.node_children = nn.ModuleList([mod5])
mod4.node_parents = nn.ModuleList([mod2, mod3])
mod5.node_parents = nn.ModuleList([mod1, mod4])


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