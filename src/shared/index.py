from typing import Tuple 
import random

class Index:
	def __init__(self, index: int =0) -> None:
		self._index = index
	def get_shuffled(self, bounds: Tuple[int, int] | int, salt: int = 0) -> int:
		if isinstance(bounds, int):
			bounds = (0, bounds)
		elif bounds[0] > bounds[1]:
			bounds = (bounds[1], bounds[0])
		#TODO: may need to change this, so that the bounding is done after the num is generated?
		return random.Random(self._index + salt).randint(*bounds)
	def get(self) -> int:
		return self._index
