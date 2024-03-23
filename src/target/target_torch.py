from __future__ import annotations

from ..shared import LockedShape
from ..schema import IRNode, ID
from ..schema.components import Component, Concat, Sum, Conv, Full, ReLU, Sigmoid, Softmax, Dropout, BatchNormalization, ChannelDropout, ReLU6
from .target_components import TargetComponents
from .python_formats import *

import itertools

class TorchComponents(TargetComponents):
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		if isinstance(component, Conv):
			#need to fix up group
			return [torch_nn_(f"Conv{len(input_shape) - 1}d({input_shape[0]}, {output_shape[0]}, {component._kernel}, {component._stride}, {component._padding}, {component._group_size}, bias=True, padding_mode='zeros')")]
		elif isinstance(component, Full):
			return [torch_nn_(f"Linear({input_shape.get_product()}, {output_shape.get_product()}, bias=True)")] 
		elif isinstance(component, ReLU):
			return [torch_nn_("ReLU()")]
		elif isinstance(component, ReLU6):
			return [torch_nn_("ReLU6()")]
		elif isinstance(component, Sigmoid):
			return [torch_nn_("Sigmoid()")]
		elif isinstance(component, Softmax):
			return [torch_nn_("Softmax(dim=1)")]
		elif isinstance(component, BatchNormalization):
			return [torch_nn_(f"BatchNorm{len(input_shape) - 1}d({input_shape[0]})")]
		elif isinstance(component, Dropout):
			return [torch_nn_(f"Dropout(p={component._p})")]
		elif isinstance(component, ChannelDropout):
			return [torch_nn_(f"ChannelDropout(p={component._p})")]
		return []
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, inputs: list[str]) -> list[str]:
		if isinstance(component, Sum):
			return [f"({' + '.join(inputs)})"]
		elif isinstance(component, Concat):
			if len(inputs) == 1:
				return [inputs[0]]
			return [torch_(f"cat({arg_list_(*inputs)}, dim=1)")]
		elif isinstance(component, Conv):
			return [view_(inputs[0], input_shape)]
		return [inputs[0]]

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
	for i, node in enumerate(ir):
		component_inits: list[str] = list(itertools.chain.from_iterable(target_components.get_init(component, node.input_shape, node.output_shape) for component in node.schema_node.get_components()))
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
		if node.id not in children_counts: #dont need to worry about register being reclaimed
			return_registers.append(register_out)
		forward_statment: str = ""
	return torch_module_(name, init_statements, to_str_list(arg_registers), forward_statements)

def view_(expr: str, shape: LockedShape) -> str:
	return f"{expr}.view(-1, {arg_list_(*to_str_list(iter(shape)))})"
def flatten_view_(expr: str, size: int | LockedShape) -> str:
	return f"{expr}.view(-1, {size if isinstance(size, int) else size.get_product()})"
def sum_(exprs: list[str]) -> str:
	return f"({' + '.join(exprs)})"
def cat_(exprs: list[str]) -> str:
	if len(exprs) == 1:
		return exprs[0]
	return torch_(f"cat({arg_list_(*exprs)}, dim=1)")
def import_torch_() -> str:
	return "import torch"
def torch_(expr: str) -> str:
	return f"torch.{expr}"
def torch_nn_(expr: str) -> str:
	return f"torch.nn.{expr}"
def torch_module_(name: str, init_statements: list[str], forward_args: list[str], forward_statments: list[str]) -> str:
	return concat_lines_(*([import_torch_()] + class_(name, [torch_nn_("Module")], 
		function_("__init__", ["self"],["super().__init__()"] + [import_torch_()] + init_statements) +
		function_("forward", ["self"] + forward_args, [import_torch_()] + forward_statments))))
