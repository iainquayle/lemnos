from __future__ import annotations

from ..shared import LockedShape

from typing import Callable
from abc import ABC as Abstract, abstractmethod

class TargetComponents(Abstract):
	def __init__(self, additional_mapping: dict[type, Callable] | None = None):
		self.mapping = additional_mapping if additional_mapping is not None else {} 
	def view(self, expr: str, shape: LockedShape) -> str:
		raise NotImplementedError
	def sum(self, *exprs: str) -> str:
		raise NotImplementedError
	def cat(self, *exprs: str) -> str:
		raise NotImplementedError
	def conv_init(self, shape_in: LockedShape, shape_out: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], group: int) -> list[str]:
		raise NotImplementedError
	def full_init(self, shape_in: LockedShape, shape_out: LockedShape) -> list[str]:
		raise NotImplementedError
	def relu_init(self) -> list[str]:
		raise NotImplementedError
	def relu6_init(self) -> list[str]:
		raise NotImplementedError
	def softmax_init(self) -> list[str]:
		raise NotImplementedError
	def sigmoid_init(self) -> list[str]:
		raise NotImplementedError
	def batch_norm_init(self, shape_in: LockedShape) -> list[str]:
		raise NotImplementedError
	def dropout_init(self, p: float) -> list[str]:
		raise NotImplementedError
	def channel_dropout_init(self, p: float, shape_in: LockedShape) -> list[str]:
		raise NotImplementedError
	def conv_forward(self, expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [expr]
	def full_forward(self, expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [expr]
	def relu_forward(self, expr: str) -> list[str]:
		return [expr]
	def relu6_forward(self, expr: str) -> list[str]:
		return [expr]
	def softmax_forward(self, expr: str) -> list[str]:
		return [expr]
	def sigmoid_forward(self, expr: str) -> list[str]:
		return [expr]
	def batch_norm_forward(self, expr: str, input_shape: LockedShape) -> list[str]:
		return [expr]
	def dropout_forward(self, expr: str, p: float) -> list[str]:
		return [expr]
	def channel_dropout_forward(self, expr: str, p: float, input_shape: LockedShape) -> list[str]:
		return [expr]
