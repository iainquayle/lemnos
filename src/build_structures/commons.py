from __future__ import annotations

from torch import Size
from typing import List

from abc import ABC as Abstract, abstractmethod

class MergeMethod(Abstract):
	@abstractmethod
	def validate_shapes(self, shapes: List[Size]) -> bool:
		pass
	@abstractmethod
	def conform_shape(self, shape: Size, sibling_shapes: List[Size]) -> Size:
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
	def conform_shape(self, shape: Size, sibling_shapes: List[Size]) -> Size:
		return shape
	def get_merge_string(self, registers: List[str]) -> str | None:
		return f"torch.cat([{', '.join(registers)}], dim=1)"
class Add(MergeMethod):
	def validate_shapes(self, shapes: List[Size]) -> bool:
		for shape in shapes:
			if shape != shapes[0]:
				return False
		return True
	def conform_shape(self, shape: Size, sibling_shapes: List[Size]) -> Size:
		return sibling_shapes[0] if len(sibling_shapes) > 0 else shape
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
