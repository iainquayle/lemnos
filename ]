from __future__ import annotations

from src.shared import Index
from src.schema.schema_node import SchemaNode
from src.schema.src_generation import *
from .model_node import ModelNode

from typing import List, Tuple, Dict, Type

from copy import copy

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, input_nodes: List[ModelNode], output_nodes: List[ModelNode]) -> None:
		self._input_nodes: List[ModelNode] = input_nodes 
		self._output_nodes: List[ModelNode] = output_nodes 
		self._ordered_node_cache: List[ModelNode] | None = []
	def get_ordered_nodes(self) -> List[ModelNode]:
		#TODO: change the ordering of nodes so that the input nodes are all first
		#	can reuse it for the forward pass then
		#	shouldnt change any of the caching for the actual evaluation of the model?
		if self._ordered_node_cache is not None and len(self._ordered_node_cache) > 0:
			return self._ordered_node_cache
		else:
			evaluation_tracker: Dict[ModelNode, int] = {}
			ordered_nodes: List[ModelNode] = []
			for node in self._input_nodes:
				evaluation_tracker[node] = 0
			evaluated_node: bool = True
			while evaluated_node:
				evaluated_node = False
				for node, visits in list(evaluation_tracker.items()):
					if visits == len(node.get_parents()):
						del evaluation_tracker[node]
						ordered_nodes.append(node)
						evaluated_node = True
						for child in node.get_children():
							if child in evaluation_tracker:
								evaluation_tracker[child] += 1
							else:
								evaluation_tracker[child] = 1
			self._ordered_node_cache = ordered_nodes
			return copy(ordered_nodes) 
	def get_index_list(self) -> List[Index]:
		return [node._index for node in self.get_ordered_nodes()]
	def get_index_schema_list(self) -> List[Tuple[Index, SchemaNode]]:
		return [(node._index, node.get_schema_node()) for node in self.get_ordered_nodes()]
	def get_torch_module_src(self, name: str) -> str:
		forward_data: List[Tuple[ModelNode, int, List[int]]] = []
		output_registers: List[int] = []
		evaluation_tracker: Dict[ModelNode, List[int]] = {} #node and the registers it is using
		register_commitments: Dict[int, int] = {} #register and nodes it still needs to be used for
		available_registers: List[int] = []
		register_count = len(self._input_nodes)
		for i, node in enumerate(self._input_nodes):
			evaluation_tracker[node] = [i]
			register_commitments[i] = 1
			evaluated_node: bool = True 
			while evaluated_node:
				evaluated_node = False
				for node, registers_in in list(evaluation_tracker.items()):
					if len(node.get_parents()) <= len(registers_in):
						del evaluation_tracker[node]
						evaluated_node = True
						for register in registers_in:
							register_commitments[register] -= 1
							if register_commitments[register] == 0:
								available_registers.append(register)
						register_out = -1
						if len(available_registers) > 0:
							register_out = available_registers.pop()
						else:
							register_out = register_count
							register_count += 1
						for child in node.get_children():
							if child in evaluation_tracker:
								evaluation_tracker[child].append(register_out)
							else:
								evaluation_tracker[child] = [register_out]
							if register_out in register_commitments:
								register_commitments[register_out] += 1
							else:
								register_commitments[register_out] = 1
						if len(node.get_children()) == 0:
							output_registers.append(register_out)
							register_commitments[register_out] = 1
						forward_data.append((node, register_out, registers_in))
		init_statements: List[str] = []
		forward_statements: List[str] = []
		def format_component(node: int, component: int | None = None) -> str:
			if component is None:
				return f"c{node}"
			else:
				return f"c{node}_{component}"
		def format_register(register: int) -> str:
			return f"r{register}"
		def format_registers(registers: List[int]) -> List[str]:
			return [format_register(r) for r in registers]
		for i, (node, register_out, registers_in) in enumerate(forward_data):
			forward_statment: str = node.get_schema_node().get_merge_method().get_merge_src(format_registers(registers_in))
			inits = node.get_inits_src()
			if len(inits) > 0:
				components = [format_component(i, j) for j in range(len(inits))]
				for component, init in zip(components, inits):
					init_statements.append(self_(assign_(component, init)))
				forward_statment = node.get_mould_view_src(forward_statment)
				for component in components:
					forward_statment = self_(call_(component, forward_statment))
				forward_statment = node.get_output_view_src(forward_statment)
			if len(node.get_children()) == 0:
				forward_statment = node.get_final_view_shape(forward_statment) 
			forward_statements.append(assign_(format_register(register_out), forward_statment))
		forward_statements.append(return_(*format_registers(output_registers)))
		src = torch_module_(name, init_statements, format_registers(list(range(len(self._input_nodes)))), forward_statements)
		return src 
	def get_torch_module_handle(self, name: str) -> Type:
		exec(self.get_torch_module_src(name))
		return eval(name)
