from __future__ import annotations

from ...shared import LockedShape

from abc import ABC as Abstract 

from typing import Any


class Component(Abstract):
	def get_statements(self, input_shape: LockedShape, output_shape: LockedShape, **kwargs) -> Any:
		raise NotImplementedError(f"must bind a statements generator for {self.__class__.__name__}")
