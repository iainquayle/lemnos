from __future__ import annotations

from ..shared import LockedShape
from ..schema import IRNode, ID
from ..schema.components import Component, Concat, Sum, Conv, Full, ReLU, Sigmoid, Softmax, Dropout, BatchNormalization, ChannelDropout, ReLU6
from ..format.format_torch import * 

from abc import ABC as Abstract, abstractmethod

import itertools

class TorchComponentFormatter(Abstract):
	@abstractmethod
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> str:
		pass
	@abstractmethod
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, input_exprs: list[str]) -> str:
		pass


class DefaultComponentFormatter(TorchComponentFormatter):
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> str:
		if isinstance(component, Conv):
			return conv_init_(input_shape, output_shape, component.get_kernel(input_shape),
				component.get_stride(input_shape), component.get_padding(input_shape),
				component.get_groups(output_shape))
		elif isinstance(component, Full):
			return full_init_(input_shape, output_shape)
		elif isinstance(component, ReLU):
			return relu_init_()
		elif isinstance(component, ReLU6):
			return relu6_init_()
		elif isinstance(component, Sigmoid):
			return sigmoid_init_() 
		elif isinstance(component, Softmax):
			return softmax_init_() 
		elif isinstance(component, BatchNormalization):
			return batchnorm_init_(output_shape) 
		elif isinstance(component, Dropout):
			return dropout_init_(component.get_probability()) 
		elif isinstance(component, ChannelDropout):
			return channeldropout_init_(component.get_probability()) 
		return "" 
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, input_exprs: list[str]) -> str:
		if isinstance(component, Sum):
			return sum_(input_exprs)
		elif isinstance(component, Concat):
			if len(input_exprs) == 1:
				return input_exprs[0]
			return cat_(input_exprs)
		else: 
			if isinstance(component, Conv):
				pass
		return input_exprs[0]

def _register_name(register: ID) -> str:
	return f"r{register}"
def _component_name(node_id: ID, component_index: int) -> str:
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
