from __future__ import annotations


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
	def inside(self, i: int | float) -> bool:
		return i >= self.lower and i <= self.upper	
	def from_ratio(self, ratio: float) -> float:
		return self.lower + ratio * self.difference()
class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.set_index(index)
	def set_index(self, index: int) -> None:
		self.index = index % Index.MAX_INDEX
	def as_ratio(self) -> float:
		return self.index / Index.MAX_INDEX