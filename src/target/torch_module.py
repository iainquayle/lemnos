from __future__ import annotations 

from ..schema import IRNode, ID
from .torch_components import torch_module_, TorchComponents
from .python_formats import *

def _get_register(max_register: ID, available_registers: list[ID]) -> tuple[ID, ID]:
	return (max_register, available_registers.pop()) if available_registers else (max_register + 1, max_register + 1)
def _format_register(register: ID) -> str:
	return f"r{register}"
def _format_component(node_id: ID, component_index: int) -> str:
	return f"c{node_id}_{component_index}"
def generate_torch_module(name: str, ir: list[IRNode]) -> str:
	target_components = TorchComponents()
	children_counts: dict[ID, int] = {}
	for node in ir:
		for parent_id in node.parent_ids:
			children_counts[parent_id] = children_counts.get(parent_id, 0) + 1
	input_registers: list[ID] = []
	output_registers: list[ID] = []
	init_statements: list[str] = []
	forward_statements: list[str] = []
	available_registers: list[ID] = []
	max_register: ID = 0
	for node in ir:
		component_inits: list[str] = node.schema_node.get_inits_src(target_components, node.input_shape, node.output_shape)
		for i, init in enumerate(component_inits):
			init_statements.append(assign_(self_(_format_component(node.id, i)), init))
		registers: list[ID] = []
		if len(node.parent_ids) == 0:
			max_register += 1
			registers = [max_register] 
		else:
			max_register, register = _get_register(max_register, available_registers)
			registers = [register]


	return torch_module_(name, init_statements, to_str_list(input_registers), forward_statements)
