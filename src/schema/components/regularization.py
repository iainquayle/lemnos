from __future__ import annotations

from ...shared import LockedShape
from ...target import TargetComponents

from abc import ABC as Abstract, abstractmethod

class Regularization(Abstract):
	@abstractmethod	
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		pass

class Dropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.dropout(self._p)
class ChannelDropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.channel_dropout(self._p, shape_in)
class BatchNormalization(Regularization):
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.batch_norm(shape_in)
