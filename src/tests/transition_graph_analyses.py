from kontrol.transitions import Transition, ConvTransition
from structures.graph import Node
from structures.commons import MergeMethod
from copy import copy
#import torch.nn as nn
#import torch.optim as optim
#import torch

print("Hello World!")


t1 = Transition()
t2 = Transition()
t3 = Transition()
t4 = Transition()
t5 = Transition()

t1.add_next_state_group({t2: True})
t2.add_next_state_group({t2: True, t3: True})
t3.add_next_state_group({t2: True, t4: True})
t4.add_next_state_group({t2: True})
t2.add_next_state_group({t5: True})


#this may be able to be distributed into adding transitions
#t1.analyse_visits()
print(t1.visits)
print(t2.visits)
print(t3.visits)
print(t4.visits)
print(t5.visits)