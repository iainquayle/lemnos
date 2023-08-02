from kontrol.transitions import Transition, ConvTransition
from structures.graph import Node
from structures.commons import MergeMethod
import torch.nn as nn
import torch

print("Hello World!")
mod1 = Node(function=nn.Linear(10, 10), shape_in=[10], shape_out=[10], merge_method=MergeMethod.ADD)
mod2 = Node(function=nn.Linear(10, 10), shape_in=[10], shape_out=[10], merge_method=MergeMethod.ADD)
mod3 = Node(function=nn.Linear(10, 10), shape_in=[10], shape_out=[10], merge_method=MergeMethod.ADD)

mod1.children = [mod2, mod3]
mod2.children = [mod3]
mod2.parents = [mod1]
mod3.parents = [mod1, mod2]

print(mod1(torch.randn(10)))