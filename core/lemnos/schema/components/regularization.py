from __future__ import annotations

from .component import Component

from abc import ABC as Abstract 


class Regularization(Component, Abstract):
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


class LayerNorm(Regularization):
	pass


class RmsNorm(Regularization):
	pass
