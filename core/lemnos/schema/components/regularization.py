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

	def __init__(self, momentum: float | None = 0.1) -> None:
		self._momentum: float | None = momentum

	def get_momentum(self) -> float | None:
		return self._momentum


class GroupNorm(Regularization):

	def __init__(self, groups: int) -> None:
		raise NotImplementedError
		self._groups: int = groups


class LayerNorm(Regularization):
	pass


class RmsNorm(Regularization):
	pass
