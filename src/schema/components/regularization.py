from __future__ import annotations

from ...shared import LockedShape
from ...target.target_components import TargetComponents
from .component import Component

from abc import ABC as Abstract, abstractmethod

class Regularization(Component, Abstract):
	pass

class Dropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.dropout(self._p)]
class ChannelDropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.channel_dropout(self._p, input_shape)]
class BatchNormalization(Regularization):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.batch_norm(input_shape)] 
