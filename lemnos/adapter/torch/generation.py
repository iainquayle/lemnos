from __future__ import annotations

from ...shared import LockedShape, ID
from ...schema import IRNode 
from ...schema.components import Component
from ...templates.torch import * 
from ...templates.python import *
from torch.nn import Module

from abc import ABC as Abstract 
from enum import Enum

from dataclasses import dataclass

from typing import Type, Callable, TypeVar 

class ShapeView(Enum):
	FLAT = 'flat' 
	REAL = 'real'
	EITHER = 'either' 

class InitType(Enum):
	CALLABLE = 'callable'
	DATA = 'data'


class IdentifierGenerator: 
	def __init__(self, namespace: str, is_member: bool):
		self._namespace = namespace
		self._identifiers: dict[str, int] = {} 
		self._is_member = is_member
	def get_identifier(self, name: str = '') -> str:
		if name not in self._identifiers:
			self._identifiers[name] = len(self._identifiers)
		identifier = f"{self._namespace}_{self._identifiers[name]}"
		return self_(identifier) if self._is_member else identifier


@dataclass(frozen=True)
class StatementGeneratorOutput:
	init_statements: list[str]
	intermediate_forward_statements: list[str]
	return_forward_expression: str

@dataclass(frozen=True)
class StatementGeneratorArgs:
	input_shape: LockedShape
	output_shape: LockedShape
	member_identifier_generator: IdentifierGenerator
	intermediate_identifier_generator: IdentifierGenerator
	input_register: str #could techinally make this an expression

ComponentType = TypeVar('ComponentType', bound=Component)

class SourceGenerator(Abstract):
	def __init__(self, generator_map: dict[Type, Callable[[ComponentType, StatementGeneratorArgs], StatementGeneratorOutput]]):
		self._generator_map = generator_map 
	def set_generator(self, component_type: Type, generator: Callable[[ComponentType, StatementGeneratorArgs], StatementGeneratorOutput]) -> None:
		no_type_generator: Any = generator
		self._generator_map[component_type] = no_type_generator
	def generate_source(self, name: str, ir: list[IRNode], add_debug_logs: bool = False) -> str:
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
			forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape) if current_shape == ShapeView.REAL else forward_statement[0])))
			#forward_statements.append(assign_(_register_name(register_out), (flatten_view_(forward_statement[0], node.output_shape))) + " # " + node.schema_node.debug_name)
			#forward_statements.append(print_(arg_list_(f"'{node.schema_node.debug_name}'", _register_name(register_out) + ".shape")))
		forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
		return concat_lines_(import_torch_(), *module_(name, [line for module_src in self.get_class_definitions(ir) for line in module_src], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))
	def create_module(self, name: str, ir: list[IRNode]) -> Module:
		source = self.generate_source(name, ir)
		exec(source)
		return locals()[name]()
	def create_module(self, name: str, ir: list[IRNode]) -> Module:
		source = generate_source(name, ir, component_formatter)
		exec(source)
		return locals()[name]()



def _register_name(register: ID) -> str:
	return f"r{register:04x}"
def _component_name(node_id: ID, component_index: int) -> str:
	return f"c{node_id:04x}_{component_index}"



