from __future__ import annotations

from abc import ABC as Abstract 

class Regularization(Abstract):
	pass

class Dropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
class ChannelDropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
class BatchNormalization(Regularization):
	pass
