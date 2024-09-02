from __future__ import annotations

from abc import ABC as Abstract 

class Component(Abstract):
	def get_forward_statements(self) -> list[str]:
		raise NotImplementedError(f"get_forwards not implemented for {self.__class__.__name__}")
	def get_backward_statements(self) -> list[str]: 
		raise NotImplementedError(f"get_backwards not implemented for {self.__class__.__name__}")
