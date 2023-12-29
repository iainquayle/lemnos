from __future__ import annotations

import torch
from torch import Size
from typing import List

from abc import ABC as Abstract, abstractmethod

#TODO:
#	make merge methods use viewing in 1d to join
class MergeMethod(Abstract):
	@abstractmethod
	def validate_shapes(self, shapes: List[Size]) -> bool:
		pass
	@abstractmethod
	def get_total_merged_shape(self, shapes: List[Size]) -> Size:
		pass
	@abstractmethod
	def get_required_shape(self, shapes: List[Size]) -> Size | None:
		pass
	@abstractmethod
	def get_merge_string(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def validate_shapes(self, shapes: List[Size]) -> bool:
		for shape in shapes:
			if shape[1:] != shapes[0][1:]:
				return False
		return True
	def get_total_merged_shape(self, shapes: List[Size]) -> Size:
		first_dim = sum([shape[0] for shape in shapes])
		return Size([first_dim] + list(shapes[0][1:]))
	def get_required_shape(self, shapes: List[Size]) -> Size | None:
		return None
	def get_merge_string(self, registers: List[str]) -> str | None:
		return f"torch.cat([{', '.join(registers)}], dim=1)"
class Add(MergeMethod):
	def validate_shapes(self, shapes: List[Size]) -> bool:
		for shape in shapes:
			if shape != shapes[0]:
				return False
		return True
	def get_total_merged_shape(self, shapes: List[Size]) -> Size:
		return shapes[0] 
	def get_required_shape(self, shapes: List[Size]) -> Size | None:
		return shapes[0]
	def get_merge_string(self, registers: List[str]) -> str | None:
		return f"{' + '.join(registers)}"
	
class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.set_index(index)
	def set_index(self, index: int) -> None:
		self.index = index % Index.MAX_INDEX
	def get_index(self, max_index: int) -> int:
		return self.index % max_index if max_index > 0 else 0
	def as_ratio(self) -> float:
		return self.index / Index.MAX_INDEX

class Bound:
	def __init__(self, lower: int | float =1, upper: int | float =1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		self.upper: int | float = upper
		self.lower: int | float = lower 
	def difference(self) -> int | float:
		return self.upper - self.lower
	def average(self) -> int | float:
		return (self.upper + self.lower) / 2
	def __contains__(self, i: int | float) -> bool:
		return i >= self.lower and i <= self.upper	
	def from_index(self, index: Index, size: int) -> int:
		return int((self.lower * size) + index.get_index((int)(self.difference() * size)))
