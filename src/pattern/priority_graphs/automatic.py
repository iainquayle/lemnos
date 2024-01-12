from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from src.model_structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any, Tuple, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy


#general idea of how priority works, is that 
# priority is higher when node should be later
# a node that joins will have the highest previous nodes priority, plus 1
# if a node doesnt make any mergers it will have a priority of 0

#this works for simple tail recursion, but whether something built off of this base can ever be made to work is not yet known