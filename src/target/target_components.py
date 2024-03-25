from __future__ import annotations

from ..shared import LockedShape
from ..schema.components import Component

from abc import ABC as Abstract, abstractmethod

class TargetComponents(Abstract):
	@abstractmethod
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		pass
	@abstractmethod
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, input_exprs: list[str]) -> list[str]:
		pass
