from __future__ import annotations

from abc import ABC as Abstract 

class Regularization(Abstract):
	pass

class Dropout(Regularization):
	def __init__(self, probability: float) -> None:
		self._probability: float = probability 
	def get_probability(self) -> float:
		return self._probability
class ChannelDropout(Regularization):
	def __init__(self, probability: float) -> None:
		self._probability: float = probability 
	def get_probability(self) -> float:
		return self._probability
class BatchNorm(Regularization):
	pass
