from ...shared import LockedShape, ID
from ...schema import IRNode 
from ...schema.components import *
from ...templates.torch import * 
from ...templates.python import *
from torch.nn import Module

from abc import ABC as Abstract, abstractmethod
from enum import Enum

class ShapeView(Enum):
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
	def get_shape_requirment(self, component: Component) -> ShapeView:
		pass
	@abstractmethod
	def get_class_definitions(self, ir: list[IRNode]) -> list[list[str]]:
		pass
	@abstractmethod
	def get_class_definition(self, ir_node: IRNode) -> list[str]:
		pass
	def generate_source(self, name: str, ir: list[IRNode]) -> str:
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
		greatest_register: ID = ID(0) 
		for node in ir:
			registers_in: list[ID] = []
			register_out: ID
			if len(node.parent_ids) == 0:
				greatest_register += 1
				node_register[node.id] = greatest_register
				registers_in = [greatest_register]
				register_out = greatest_register
				forward_statements.append(assign_(_register_name(register_out), flatten_view_(_register_name(greatest_register), node.input_shape)))
				arg_registers.append(greatest_register)
			else:
				for id in node.parent_ids:
					registers_in.append(node_register[id])
					children_counts[id] -= 1
					if children_counts[id] == 0:
						available_registers.append(node_register[id])
				if len(available_registers) == 0:
					greatest_register += 1
					register_out = greatest_register
				else:
					register_out = available_registers.pop()
			node_register[node.id] = register_out
			if node.id not in children_counts: #dont need to worry about register being reclaimed
				return_registers.append(register_out)
			forward_statement = [_register_name(register) for register in registers_in] 
			current_shape = ShapeView.FLAT
			for i, component in enumerate(node.schema_node.get_components()):
				if (init := self.get_init(component, node.input_shape, node.output_shape)) != "":
					init_statements.append(assign_(self_(_component_name(node.id, i)), self_(init)) + " # " + str(node.schema_node.debug_name))
				if self.get_shape_requirment(component) == ShapeView.REAL and current_shape == ShapeView.FLAT:
					forward_statement = [view_(expr, node.input_shape) for expr in forward_statement]
				elif self.get_shape_requirment(component) == ShapeView.FLAT and current_shape == ShapeView.REAL:
					forward_statement = [flatten_view_(expr, node.input_shape) for expr in forward_statement]
				current_shape = self.get_shape_requirment(component)
				forward_statement = [self.get_forward(component, node.input_shape, node.output_shape, self_(_component_name(node.id, i)), forward_statement)]
			#forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape) if current_shape == ShapeView.REAL else forward_statement[0])))
			forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape))) + " # " + node.schema_node.debug_name)
			#forward_statements.append(print_(arg_list_(f"'{node.schema_node.debug_name}'", _register_name(register_out) + ".shape")))
		forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
		return concat_lines_(import_torch_(), *module_(name, [line for module_src in self.get_class_definitions(ir) for line in module_src], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))

class DefaultComponentFormatter(TorchComponentFormatter):
	def get_init(self, component: Component, input_shape: LockedShape, output_shape: LockedShape) -> str:
		if isinstance(component, Conv):
			return conv_init_(input_shape, output_shape, component.get_kernel(input_shape),
				component.get_stride(input_shape), component.get_padding(input_shape),
				component.get_dilation(input_shape), component.get_groups(input_shape, output_shape), isinstance(component, MixedConv))
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
		elif isinstance(component, SiLU):
			return silu_init_()
		elif isinstance(component, BatchNorm):
			return batchnorm_init_(output_shape) 
		elif isinstance(component, LayerNorm):
			return layernorm_init_(output_shape) 
		elif isinstance(component, Dropout):
			return dropout_init_(component.get_probability()) 
		elif isinstance(component, ChannelDropout):
			return channeldropout_init_(input_shape, component.get_probability()) 
		elif isinstance(component, GLU):
			return glu_init_()
		return "" 
	def get_forward(self, component: Component, input_shape: LockedShape, output_shape: LockedShape, component_name: str, input_exprs: list[str]) -> str:
		if isinstance(component, Sum):
			return sum_(input_exprs)
		elif isinstance(component, Concat):
			if len(input_exprs) == 1:
				return input_exprs[0]
			return self_(cat_(input_exprs))
		return call_(component_name, *input_exprs)
	def get_shape_requirment(self, component: Component) -> ShapeView:
		if isinstance(component, Conv):
			return ShapeView.REAL
		elif isinstance(component, BatchNorm):
			return ShapeView.REAL
		elif isinstance(component, LayerNorm):
			return ShapeView.REAL
		elif isinstance(component, ChannelDropout):
			return ShapeView.REAL
		elif isinstance(component, Full):
			return ShapeView.FLAT
		elif isinstance(component, Concat) or isinstance(component, Sum):
			return ShapeView.FLAT
		return ShapeView.EITHER
	def get_class_definitions(self, ir: list[IRNode]) -> list[list[str]]:
		definitions = []
		for node in ir:
			new_definition = self.get_class_definition(node)
			if new_definition not in definitions:
				definitions.append(new_definition)
		return definitions
	def get_class_definition(self, ir_node: IRNode) -> list[str]:
		if isinstance(ir_node.schema_node.get_transform(), MixedConv):
			return conv_mix_definition_(ir_node.input_shape.dimensionality() - 1)
		return [""]

def _register_name(register: ID) -> str:
	return f"r{register:04x}"
def _component_name(node_id: ID, component_index: int) -> str:
	return f"c{node_id:04x}_{component_index}"
def generate_source(name: str, ir: list[IRNode], component_formatter: TorchComponentFormatter = DefaultComponentFormatter()) -> str:
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
	greatest_register: ID = ID(0) 
	for node in ir:
		registers_in: list[ID] = []
		register_out: ID
		if len(node.parent_ids) == 0:
			greatest_register += 1
			node_register[node.id] = greatest_register
			registers_in = [greatest_register]
			register_out = greatest_register
			forward_statements.append(assign_(_register_name(register_out), flatten_view_(_register_name(greatest_register), node.input_shape)))
			arg_registers.append(greatest_register)
		else:
			for id in node.parent_ids:
				registers_in.append(node_register[id])
				children_counts[id] -= 1
				if children_counts[id] == 0:
					available_registers.append(node_register[id])
			if len(available_registers) == 0:
				greatest_register += 1
				register_out = greatest_register
			else:
				register_out = available_registers.pop()
		node_register[node.id] = register_out
		if node.id not in children_counts: #dont need to worry about register being reclaimed
			return_registers.append(register_out)
		forward_statement = [_register_name(register) for register in registers_in] 
		current_shape = ShapeView.FLAT
		for i, component in enumerate(node.schema_node.get_components()):
			if (init := component_formatter.get_init(component, node.input_shape, node.output_shape)) != "":
				init_statements.append(assign_(self_(_component_name(node.id, i)), self_(init)) + " # " + str(node.schema_node.debug_name))
			if component_formatter.get_shape_requirment(component) == ShapeView.REAL and current_shape == ShapeView.FLAT:
				forward_statement = [view_(expr, node.input_shape) for expr in forward_statement]
			elif component_formatter.get_shape_requirment(component) == ShapeView.FLAT and current_shape == ShapeView.REAL:
				forward_statement = [flatten_view_(expr, node.input_shape) for expr in forward_statement]
			current_shape = component_formatter.get_shape_requirment(component)
			forward_statement = [component_formatter.get_forward(component, node.input_shape, node.output_shape, self_(_component_name(node.id, i)), forward_statement)]
		#forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape) if current_shape == ShapeView.REAL else forward_statement[0])))
		forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape))) + " # " + node.schema_node.debug_name)
		#forward_statements.append(print_(arg_list_(f"'{node.schema_node.debug_name}'", _register_name(register_out) + ".shape")))
	forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
	return concat_lines_(import_torch_(), *module_(name, [line for module_src in component_formatter.get_class_definitions(ir) for line in module_src], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))
def create_module(name: str, ir: list[IRNode], component_formatter: TorchComponentFormatter = DefaultComponentFormatter()) -> Module:
	source = generate_source(name, ir, component_formatter)
	exec(source)
	return locals()[name]()
