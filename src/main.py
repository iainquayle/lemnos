from kontrol.transitions import Transition, ConvTransition
from structures.tree import Node
from structures.commons import MergeMethod
from tests import graph_features, tree_features, transition_graph_analyses
from copy import copy
import torch.nn as nn
import torch.optim as optim
import torch

print("Hello World!")

transition_graph_analyses.test_cylical()
