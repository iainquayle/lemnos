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

from typing import Type, Callable 

class ShapeView(Enum):
	FLAT = 'flat' 
	REAL = 'real'
	EITHER = 'either' 

class InitType(Enum):
	CALLABLE = 'callable'
	CONST = 'const'


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
	shape_view: ShapeView

@dataclass(frozen=True)
class StatementGeneratorArgs:
	input_shape: LockedShape
	output_shape: LockedShape
	member_identifier_generator: IdentifierGenerator
	intermediate_identifier_generator: IdentifierGenerator
	input_registers: list[str] #may move this to expressions but that would take more optimization work to make useful

class SourceGenerator(Abstract):
	def __init__(self, generator_map: dict[Type[Component], Callable[[Any, StatementGeneratorArgs], StatementGeneratorOutput]]):
		self._generator_map = generator_map 
	def set_generator(self, component_type: Type[Component], generator: Callable[[Any, StatementGeneratorArgs], StatementGeneratorOutput]) -> None:
		no_type_generator: Any = generator
		self._generator_map[component_type] = no_type_generator
	def generate_source(self, name: str, ir: list[IRNode], add_debug_logs: bool = False) -> str:
		node_reference_count: dict[ID, int] = {}
		for node in ir:
			for parent_id in node.parent_ids:
				node_reference_count[parent_id] = node_reference_count.get(parent_id, 0) + 1

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
				greatest_register += 1 #dont need to create a new register, should be able to do the exact same thing as is done below, would just need to add it to arg_registers
				node_register[node.id] = greatest_register
				registers_in = [greatest_register]
				register_out = greatest_register
				arg_registers.append(greatest_register)
				#forward_statements.append(assign_(_register_name(register_out), flatten_view_(_register_name(greatest_register), node.input_shape)))
			else:
				for id in node.parent_ids:
					registers_in.append(node_register[id])
					node_reference_count[id] -= 1
					if node_reference_count[id] == 0:
						available_registers.append(node_register[id])
				if len(available_registers) == 0:
					greatest_register += 1
					register_out = greatest_register
				else:
					register_out = available_registers.pop()
			node_register[node.id] = register_out
			if node.id not in node_reference_count: #dont need to worry about register being reclaimed
				return_registers.append(register_out)
			
			#clean this up, yuck

			#still not correct, shape views use the output shape, but thats obviously shouldnt be the case

			shape_view = ShapeView.FLAT
			components = node.schema_node.get_components()
			component_statements: list[StatementGeneratorOutput] = []
			for i, component in enumerate(components):
				component_statement = self._generator_map[type(component)](
					component,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'c_{node.id}_{i}', True), IdentifierGenerator('i', False), list(map(_register_name, registers_in)))
				)
				if not component_statement.shape_view == shape_view:
					if component_statement.shape_view == ShapeView.REAL:
						shape_view = ShapeView.REAL
						component_statements.append(StatementGeneratorOutput([], [], view_(_register_name(registers_in[0]), node.output_shape), ShapeView.REAL))
					if component_statement.shape_view == ShapeView.FLAT:
						shape_view = ShapeView.FLAT
						component_statements.append(StatementGeneratorOutput([], [], flatten_view_(_register_name(registers_in[0]), node.output_shape), ShapeView.FLAT))
				registers_in = [register_out]
				component_statements.append(component_statement)
			if shape_view == ShapeView.REAL:
				component_statements.append(StatementGeneratorOutput([], [], flatten_view_(_register_name(registers_in[0]), node.output_shape), ShapeView.FLAT))

			for statements in component_statements:
				init_statements.extend(statements.init_statements)

				forward_statements.extend(statements.intermediate_forward_statements)
				forward_statements.append(assign_(_register_name(register_out), statements.return_forward_expression) + f"{'	' * 2}# ID: {node.id} : {node.schema_node.debug_name}")
		forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
		return concat_lines_(import_torch_(), *module_(name, [], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))
	def create_module(self, name: str, ir: list[IRNode]) -> Module:
		source = self.generate_source(name, ir)
		exec(source)
		return locals()[name]()



def _register_name(register: ID) -> str:
	return f"r{register:04x}"



