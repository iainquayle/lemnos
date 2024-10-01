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

@dataclass(frozen=True)
class ComponentStatements:
	init_statements: list[str]
	intermediate_forward_statements: list[str]
	return_forward_expression: str

@dataclass(frozen=True)
class StatementGenerator:
	generator: Callable[[Any, StatementGeneratorArgs], ComponentStatements]
	required_view: ShapeView

@dataclass(frozen=True)
class StatementGeneratorArgs:
	input_shape: LockedShape
	output_shape: LockedShape
	member_identifier_generator: IdentifierGenerator
	intermediate_identifier_generator: IdentifierGenerator
	input_registers: list[str] #may move this to expressions but that would take more optimization work to make useful

@dataclass(frozen=True)
class _Register:
	id: ID
	shape: LockedShape
	view: ShapeView

class SourceGenerator(Abstract):
	def __init__(self, generator_map: dict[Type[Component], StatementGenerator] = {}):
		self._generator_map = generator_map 
	def set_generator(self, component_type: Type[Component], generator: StatementGenerator) -> None:
		no_type_generator: Any = generator
		self._generator_map[component_type] = no_type_generator
	def generate_source(self, name: str, ir: list[IRNode], add_debug_logs: bool = False) -> str:
		node_reference_count: dict[ID, int] = {}
		for node in ir:
			for parent_id in node.parent_ids:
				node_reference_count[parent_id] = node_reference_count.get(parent_id, 0) + 1

		node_output_registers: dict[ID, _Register] = {}
		arg_registers: list[ID] = []
		return_registers: list[ID] = []

		init_statements: list[str] = []
		forward_statements: list[str] = []

		available_registers: list[ID] = []
		greatest_register: ID = ID(0) 

		for node in ir:
			registers_in: list[_Register] = []
			register_out_id: ID = ID(0) 
			if len(node.parent_ids) == 0:
				greatest_register += 1 #not necessary to create a new register, should be able to do the exact same thing as is done below, would just need to add it to arg_registers
				register = _Register(greatest_register, node.input_shape, ShapeView.REAL)
				node_output_registers[node.id] = register 
				registers_in = [register] 
				register_out = register
				arg_registers.append(greatest_register)
			else:
				for id in node.parent_ids:
					registers_in.append(node_output_registers[id])
					node_reference_count[id] -= 1
					if node_reference_count[id] == 0:
						available_registers.append(node_output_registers[id].id)
				if len(available_registers) == 0:
					greatest_register += 1
					register_out_id = greatest_register
				else:
					register_out_id = available_registers.pop()
			
			#the register viewing needs to be injected into the components, it cant have a side effect on the register itself
			#still need make the correct shape flow, whether that is recording all shapes in the schema solution or recreating them here


			for i, component in enumerate(node.schema_node.get_components()):

				statement_generator = self._generator_map[type(component)]
				input_expressions: list[str] = [] 
				for register in registers_in:
					base_expression = _register_name(register.id)
					if register.view == ShapeView.FLAT and statement_generator.required_view == ShapeView.REAL:
						input_expressions.append(flatten_view_(base_expression, register.shape))
					elif register.view == ShapeView.REAL and statement_generator.required_view == ShapeView.FLAT:
						input_expressions.append(view_(base_expression, register.shape))
					else:
						input_expressions.append(base_expression)
					 
				component_statements = statement_generator.generator(component, StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'c_{node.id}_{i}', True), IdentifierGenerator('i', False), input_expressions))
				#registers_in = [register_out]
				#component_statements.append(component_statement)
				raise NotImplementedError('fix this still')

				init_statements.extend(component_statements.init_statements)
				forward_statements.extend(component_statements.intermediate_forward_statements)
				forward_statements.append(assign_(_register_name(register_out_id), component_statements.return_forward_expression) + f"{'	' * 2}# ID: {node.id} : {node.schema_node.debug_name}")

			node_output_registers[node.id] = registers_in[0] #if this ever breaks something is wrong in the schema compilation 
			if node.id not in node_reference_count: #dont need to worry about register being reclaimed
				return_registers.append(register_out_id)
		forward_statements.append(return_(*[_register_name(register) for register in return_registers]))
		return concat_lines_(import_torch_(), *module_(name, [], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))
	def create_module(self, name: str, ir: list[IRNode]) -> Module:
		source = self.generate_source(name, ir)
		exec(source)
		return locals()[name]()



def _register_name(register: ID) -> str:
	return f"r{register:04x}"



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
