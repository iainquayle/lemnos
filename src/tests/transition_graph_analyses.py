from kontrol.transitions import Transition, TransitionGroup, ConvTransition
from structures.graph import Node
from structures.commons import MergeMethod
from copy import copy
#import torch.nn as nn
#import torch.optim as optim
#import torch

t1 = Transition()
t2 = Transition()
t3 = Transition()
t4 = Transition()
t5 = Transition()

def test_basic():
	t1.add_next_state_group(TransitionGroup({t2: True, t3: True}))
	t2.add_next_state_group(TransitionGroup({t3: True}))
	print(t1.get_full_str())
	print(t2.get_full_str())
	print(t3.get_full_str())

def test_cylical():
	t1.add_next_state_group(TransitionGroup({t2: True}))
	t2.add_next_state_group(TransitionGroup({t1: True, t2: True}))
	t2.add_next_state_group(TransitionGroup({t3: True}))
	print(t1.get_full_str())
	print(t2.get_full_str())
	print(t3.get_full_str())