from __future__ import annotations

from abc import ABC as Abstract, abstractmethod
from src.shared.shape import LockedShape

from typing import Tuple

#src generation responsibilities
#	model
#		registers
#		assigning 
#	schema node
#		eval
#		view
#	component
#		init

class ComponentSrc(Abstract):
	def get_component_init_src(self, shape_in: LockedShape, shape_out: LockedShape, id: int) -> str:
		return f"{self.get_component_name_src(id)} = {self._get_component_init_src(shape_in, shape_out)}"
	@abstractmethod
	def get_component_name_src(self, id: int) -> str:
		pass
	@abstractmethod
	def _get_component_init_src(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		pass

def unroll_(*values: str) -> str:
	return f"{', '.join(values)}"
def assign_(name: str, value: str) -> str:
	return f"{name} = {value}"
def self_(name: str) -> str:
	return f"self.{name}"
def return_(*values: str) -> str:
	return f"return {unroll_(*values)}"
def call_(name: str, *args: str) -> str:
	return f"{name}({unroll_(*args)})"
def sum_(*values: str) -> str:
	return f"({' + '.join(values)})"
def cat_(*values: str) -> str:
	return f"torch.cat(({unroll_(*values)}), dim=1)"

def conv_(shape_in: LockedShape, shape_out: LockedShape, kernel: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...]) -> str:
	return f"Conv{len(kernel)}d({shape_in[0]}, {shape_out[0]}, {kernel}, {stride}, {padding})"
