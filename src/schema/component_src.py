from __future__ import annotations

from abc import ABC as Abstract, abstractmethod
from src.shared.shape import LockedShape

class ComponentSrc(Abstract):
	def get_component_init_src(self, shape_in: LockedShape, shape_out: LockedShape, id: int) -> str:
		return f"{self.get_component_name_src(id)} = {self._get_component_init_src(shape_in, shape_out)}"
	@abstractmethod
	def get_component_name_src(self, id: int) -> str:
		pass
	@abstractmethod
	def _get_component_init_src(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		pass
