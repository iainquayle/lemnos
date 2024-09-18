from __future__ import annotations

from ...shared import LockedShape

from abc import ABC as Abstract 

from typing import Any, Callable


type Formatter = Callable[[Component, LockedShape, LockedShape, dict[str, Any]], Any]

class Component(Abstract):
	@classmethod
	def attach_statement_generator(cls, formatter: Formatter) -> None:
		cls.formatter = formatter
	def get_statements(self, input_shape: LockedShape, output_shape: LockedShape, **kwargs: Any) -> Any:
		raise NotImplementedError(f"must bind a statements generator for {self.__class__.__name__}")
