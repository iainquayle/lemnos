from __future__ import annotations

from ..schema.components import Component 

from abc import ABC as Abstract, abstractmethod

class Component(Abstract):
	@abstractmethod
	def get_init(self, component) -> str:
		pass
	@abstractmethod
	def get_forward(self, component) -> str:
		pass
