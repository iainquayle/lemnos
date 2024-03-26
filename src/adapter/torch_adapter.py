from __future__ import annotations

from ..shared import LockedShape
from ..schema import IRNode, ID
from ..schema.components import Component, Concat, Sum, Conv, Full, ReLU, Sigmoid, Softmax, Dropout, BatchNormalization, ChannelDropout, ReLU6
from ..format.format_torch import * 

from abc import ABC as Abstract, abstractmethod
from enum import Enum

import itertools

class ShapeRequirement(Enum):
	FLAT = 'flat' 
	REAL = 'real'
	EITHER = 'either' 

class TorchComponentFormatter(Abstract):
	@abstractmethod
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> str:
		pass
	@abstractmethod
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, component_name: str, input_exprs: list[str]) -> str:
		pass
	@abstractmethod
	def get_shape_requirment(self, component: Component) -> ShapeRequirement:
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
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, component_name: str, input_exprs: list[str]) -> str:
		if isinstance(component, Sum):
			return sum_(input_exprs)
		elif isinstance(component, Concat):
			if len(input_exprs) == 1:
				return input_exprs[0]
			return cat_(input_exprs)
		return call_(component_name, *input_exprs)
	def get_shape_requirment(self, component: Component) -> ShapeRequirement:
		if isinstance(component, Conv):
			return ShapeRequirement.REAL
		elif isinstance(component, BatchNormalization):
			return ShapeRequirement.REAL
		elif isinstance(component, ChannelDropout):
			return ShapeRequirement.REAL
		elif isinstance(component, Full):
			return ShapeRequirement.FLAT
		elif isinstance(component, Concat) or isinstance(component, Sum):
			return ShapeRequirement.FLAT
		return ShapeRequirement.EITHER

def _register_name(register: ID) -> str:
	return f"r{register:04x}"
def _component_name(node_id: ID, component_index: int) -> str:
	return f"c{node_id:04x}_{component_index}"
def generate_torch_module(name: str, ir: list[IRNode], component_formatter: TorchComponentFormatter = DefaultComponentFormatter()) -> str:
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
		forward_statement = [_register_name(node_register[id]) for id in node.parent_ids] 
		current_shape = ShapeRequirement.FLAT
		for i, component in enumerate(node.schema_node.get_components()):
			init_statements.append(assign_(self_(_component_name(node.id, i)), component_formatter.get_init(component, node.input_shape, node.output_shape)))
			if component_formatter.get_shape_requirment(component) == ShapeRequirement.REAL and current_shape == ShapeRequirement.FLAT:
				forward_statement = [view_(expr, node.input_shape) for expr in forward_statement]
			elif component_formatter.get_shape_requirment(component) == ShapeRequirement.FLAT and current_shape == ShapeRequirement.REAL:
				forward_statement = [flatten_view_(expr, node.input_shape) for expr in forward_statement]
			current_shape = component_formatter.get_shape_requirment(component)
			forward_statement = [component_formatter.get_forward(component, node.input_shape, node.output_shape, self_(_component_name(node.id, i)), forward_statement)]
		forward_statements.append(assign_(_register_name(register_out), flatten_view_(forward_statement[0], node.output_shape)))
	forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
	return module_(name, init_statements, to_str_list(arg_registers), forward_statements)
