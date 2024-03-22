from __future__ import annotations 

from ...schema.ir_compilation import IRNode, ID
from .torch_components import torch_module_, TorchComponents
from ..python_formats import *

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
	node_register: dict[ID, ID] = {}
	arg_registers: list[ID] = []
	return_registers: list[ID] = []
	init_statements: list[str] = []
	forward_statements: list[str] = []
	available_registers: list[ID] = []
	max_register: ID = 0
	for node in ir:
		component_inits: list[str] = node.schema_node.get_inits_src(target_components, node.input_shape, node.output_shape)
		for i, init in enumerate(component_inits):
			init_statements.append(assign_(self_(_format_component(node.id, i)), init))
		registers_in: list[ID] = []
		register_out: ID
		if len(node.parent_ids) == 0:
			max_register += 1
			node_register[node.id] = max_register
			registers_in = [max_register]
			register_out = max_register
			arg_registers.append(max_register)
		else:
			for id in node.parent_ids:
				registers_in.append(node_register[id])
				children_counts[id] -= 1
				if children_counts[id] == 0:
					available_registers.append(node_register[id])
			if len(available_registers) == 0:
				max_register += 1
				register_out = max_register
			else:
				register_out = available_registers.pop()
		node_register[node.id] = register_out
		if children_counts[node.id] == 0: #dont need to worry about register being reclaimed
			return_registers.append(register_out)
		forward_statment: str = ""
	return torch_module_(name, init_statements, to_str_list(arg_registers), forward_statements)
