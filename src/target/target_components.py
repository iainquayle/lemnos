from __future__ import annotations

from ..shared import LockedShape

from typing import Callable
from abc import ABC as Abstract, abstractmethod

class TargetComponents(Abstract):
	def __init__(self, additional_mapping: dict[type, Callable] | None = None):
		self.mapping = additional_mapping if additional_mapping is not None else {} 
	@abstractmethod
	def view(self, expr: str, shape: LockedShape) -> str:
		pass
	@abstractmethod
	def sum(self, *exprs: str) -> str:
		pass
	@abstractmethod
	def cat(self, *exprs: str) -> str:
		pass
	@abstractmethod
	def conv(self, shape_in: LockedShape, shape_out: LockedShape, kernel: tuple[int, ...], stride: tuple[int, ...], padding: tuple[int, ...], group: int) -> str:
		pass
	@abstractmethod
	def full(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		pass
	@abstractmethod
	def relu(self) -> str:
		pass
	@abstractmethod
	def relu6(self) -> str:
		pass
	@abstractmethod
	def softmax(self) -> str:
		pass
	@abstractmethod
	def sigmoid(self) -> str:
		pass
	@abstractmethod
	def batch_norm(self, shape_in: LockedShape) -> str:
		pass
	@abstractmethod
	def dropout(self, p: float) -> str:
		pass
	@abstractmethod
	def channel_dropout(self, p: float, shape_in: LockedShape) -> str:
		pass
