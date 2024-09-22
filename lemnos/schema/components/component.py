from __future__ import annotations

#from ...shared import LockedShape

from abc import ABC as Abstract 

#from typing import Any, Protocol 


#class StatementGenerator(Protocol):
#	def __call__(self: Component, input_shape: LockedShape, output_shape: LockedShape, register: str, **kwargs: Any) -> Any:
#		pass

class Component(Abstract):
	pass
	#@classmethod
	#def bind_generator(cls, generator: StatementGenerator) -> None:
	#	typed_generator: Any = generator
	#	cls.get_statements = typed_generator
	#def generate_statements(self, input_shape: LockedShape, output_shape: LockedShape, register: str, **kwargs: Any) -> Any:
	#	raise NotImplementedError(f"must bind a statements generator for {self.__class__.__name__}")
