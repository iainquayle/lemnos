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
			
			#guess more logic needed here. can add in shape tracking
			#track what register is being used for input, output doesnt need it
			#decide whether to eager or lazy flatten the shapes
			#	if on average, one to many, then eager 
			#	if on average, one to one, then lazy
			#	take u net, the cross connections induce a one to many, but really the rest are one to one
			#		as well, one of those connections doesnt require a aggregation, so doesnt need the flattened 
			#annoying thing with lazy, have to track the shape of the register...
			#not like it is going to add much overhead...

			#can just turn this into a loop? do have the get components function...


			shape_view = ShapeView.FLAT
			components = node.schema_node.get_components()
			component_statements: list[StatementGeneratorOutput] = []
			for component in components:
				component_statements.append(self._generator_map[type(component)](
					component,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'agg_{node.id}', True), IdentifierGenerator('i', False), list(map(_register_name, registers_in)))
				))
				#if not correct view, switch
				#	this needs to be figured out though, need some kind of look ahead essentially, so, instead, probably map these as such, then in the next pass deal with view
				registers_in = [register_out]

			if (aggregation := node.schema_node.get_aggregation()) is not None:
				component_statements.append(self._generator_map[type(aggregation)](
					aggregation,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'agg_{node.id}', True), IdentifierGenerator('i', False), list(map(_register_name, registers_in)))
				))
				registers_in = [register_out]
			if (transformation := node.schema_node.get_transformation()) is not None:
				component_statements.append(self._generator_map[type(transformation)](
					transformation,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'trans_{node.id}', True), IdentifierGenerator('i', False), [_register_name(register_out)])
				))
				registers_in = [register_out]
			if (activation := node.schema_node.get_activation()) is not None:
				component_statements.append(self._generator_map[type(activation)](
					activation,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'act_{node.id}', True), IdentifierGenerator('i', False), [_register_name(register_out)])
				))
				registers_in = [register_out]
			if (regularization := node.schema_node.get_regularization()) is not None:
				component_statements.append(self._generator_map[type(regularization)](
					regularization,
					StatementGeneratorArgs(node.input_shape, node.output_shape, IdentifierGenerator(f'reg_{node.id}', True), IdentifierGenerator('i', False), [_register_name(register_out)])
				))
			

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



