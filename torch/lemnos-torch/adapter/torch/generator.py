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
		return_registers: list[_Register] = []
		init_statements: list[str] = []
		forward_statements: list[str] = []
		available_registers: list[ID] = []
		greatest_register: ID = ID(0) 
		for node in ir:
			registers_in: list[_Register] = []
			register_out_id: ID = ID(0) 
			if len(node.parent_ids) == 0:
				greatest_register += 1 #not necessary to create a new register, should be able to do the exact same thing as is done below, would just need to add it to arg_registers
				register = _Register(greatest_register, node.shape_trace.get_input_shape(), ShapeView.REAL)
				node_output_registers[node.id] = register 
				registers_in = [register] 
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
			if (aggregation := node.schema_node.get_aggregation()) is not None:
				statement_generator = self._generator_map[type(aggregation)]
				input_expressions = list(map(lambda register: _view_register(register, statement_generator.required_view), registers_in))
				component_statements = statement_generator.generator(aggregation, StatementGeneratorArgs(
					node.shape_trace.aggregation_shape, 
					node.shape_trace.aggregation_shape, 
					IdentifierGenerator(f'c_{node.id}_ag', True), 
					IdentifierGenerator('i', False), 
					input_expressions))
				registers_in = [_Register(register_out_id, node.shape_trace.aggregation_shape, _get_register_view(registers_in, statement_generator))]
				init_statements.extend(component_statements.init_statements)
				forward_statements.extend(component_statements.intermediate_forward_statements)
				forward_statements.append(_format_return_forward_statement(register_out_id, component_statements, node))
			if (transformation := node.schema_node.get_transformation()) is not None:
				statement_generator = self._generator_map[type(transformation)]
				input_expressions = list(map(lambda register: _view_register(register, statement_generator.required_view), registers_in))
				component_statements = statement_generator.generator(transformation, StatementGeneratorArgs(
					node.shape_trace.aggregation_shape, 
					node.shape_trace.transformion_shape, 
					IdentifierGenerator(f'c_{node.id}_tr', True), 
					IdentifierGenerator('i', False), 
					input_expressions))
				registers_in = [_Register(register_out_id, node.shape_trace.transformion_shape, _get_register_view(registers_in, statement_generator))]
				init_statements.extend(component_statements.init_statements)
				forward_statements.extend(component_statements.intermediate_forward_statements)
				forward_statements.append(_format_return_forward_statement(register_out_id, component_statements, node))
			if (regularization := node.schema_node.get_regularization()) is not None:
				statement_generator = self._generator_map[type(regularization)]
				input_expressions = list(map(lambda register: _view_register(register, statement_generator.required_view), registers_in))
				component_statements = statement_generator.generator(regularization, StatementGeneratorArgs(
					node.shape_trace.activation_shape, 
					node.shape_trace.regularization_shape, 
					IdentifierGenerator(f'c_{node.id}_re', True), 
					IdentifierGenerator('i', False), 
					input_expressions))
				registers_in = [_Register(register_out_id, node.shape_trace.regularization_shape, _get_register_view(registers_in, statement_generator))]
				init_statements.extend(component_statements.init_statements)
				forward_statements.extend(component_statements.intermediate_forward_statements)
				forward_statements.append(_format_return_forward_statement(register_out_id, component_statements, node))
			if (activation := node.schema_node.get_activation()) is not None:
				statement_generator = self._generator_map[type(activation)]
				input_expressions = list(map(lambda register: _view_register(register, statement_generator.required_view), registers_in))
				component_statements = statement_generator.generator(activation, StatementGeneratorArgs(
					node.shape_trace.transformion_shape, 
					node.shape_trace.activation_shape, 
					IdentifierGenerator(f'c_{node.id}_ac', True), 
					IdentifierGenerator('i', False), 
					input_expressions))
				registers_in = [_Register(register_out_id, node.shape_trace.activation_shape, _get_register_view(registers_in, statement_generator))]
				init_statements.extend(component_statements.init_statements)
				forward_statements.extend(component_statements.intermediate_forward_statements)
				forward_statements.append(_format_return_forward_statement(register_out_id, component_statements, node))
			node_output_registers[node.id] = registers_in[0] #if this ever breaks something is wrong in the schema compilation 
			if node.id not in node_reference_count: #dont need to worry about register being reclaimed
				return_registers.append(registers_in[0])
		forward_statements.append(return_(*[_register_name(register.id) if register.view == ShapeView.REAL else view_(_register_name(register.id), register.shape) for register in return_registers]))
		return concat_lines_(import_torch_(), *module_(name, [], [], init_statements, list(map(_register_name, arg_registers)), forward_statements))
	def __copy__(self) -> SourceGenerator:
		return SourceGenerator(self._generator_map.copy())

	def create_module(self, name: str, ir: list[IRNode]) -> Module:
		source = self.generate_source(name, ir)
		exec(source)
		return locals()[name]()


def _get_register_view(input_registers: list[_Register], statement_generator: StatementGenerator) -> ShapeView:
	if statement_generator.required_view == ShapeView.EITHER:
		return input_registers[0].view
	return statement_generator.required_view


def _format_return_forward_statement(register_out_id: ID, component_statements: ComponentStatements, node: IRNode) -> str:
	return assign_(_register_name(register_out_id), component_statements.return_forward_expression) + f"{'	' * 2}# ID: {node.id} : {node.schema_node.debug_name}"


def _view_register(register: _Register, view: ShapeView) -> str:
	if view == ShapeView.FLAT and register.view == ShapeView.REAL:
		return flatten_view_(_register_name(register.id), register.shape)
	elif view == ShapeView.REAL and register.view == ShapeView.FLAT:
		return view_(_register_name(register.id), register.shape)
	else:
		return _register_name(register.id)


def _register_name(register: ID) -> str:
	return f"r{register:04x}"


class IdentifierGenerator: 
	def __init__(self, namespace: str, is_member: bool):
		self._namespace = namespace
		self._identifiers: dict[str, int] = {} 
		self._is_member = is_member
	def get_identifier(self, name: str | int = '') -> str:
		name = str(name)
		if name not in self._identifiers:
			self._identifiers[name] = len(self._identifiers)
		identifier = f"{self._namespace}_{self._identifiers[name]}"
		return self_(identifier) if self._is_member else identifier
