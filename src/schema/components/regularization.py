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
		return [target.dropout_init(self._p)]
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
class ChannelDropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.channel_dropout_init(self._p, input_shape)]
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
class BatchNormalization(Regularization):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.batch_norm_init(input_shape)] 
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
