from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from src.structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod
from typing import List, Set, Dict, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

class State:
	def __init__(self):
		self.transitions: List[TransitionGroup] = []
		self.priority: int = 0
	def add_next_state_group(self, group: TransitionGroup) -> None:
		self.transitions.append(copy(group))
	@abstractmethod
	def analyze(self) -> None:
		pass

class TransitionGroup:
	class TransitionData(NamedTuple):
		optional: bool = False
		join_existing: bool = False	
		def __str__(self) -> str:
			return f"({self.optional} {self.join_existing})"
		def __repr__(self) -> str:
			return str(self)
	def __init__(self, transitions: Dict[State, TransitionData] =dict())  -> None:
		self.transitions: Dict[State, TransitionGroup.TransitionData] = copy(transitions)
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)