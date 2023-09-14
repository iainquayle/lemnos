from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from src.structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any, Tuple, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy
