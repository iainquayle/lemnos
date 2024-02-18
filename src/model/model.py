from __future__ import annotations


from src.schema.schema_node import SchemaNode
from src.shared.shape import LockedShape
from src.shared.index import Index
from src.schema.src_generation import * 

from typing import List, Tuple, Iterable, Dict, Type
from typing_extensions import Self

	#tracking node evaluation progression
		#need dict of nodes, w set of parents 
		#walk through, expanding those which have full set of parents
	#tracking register use
		#need list of registers
		#need dict of nodes, w set of children, and a register
		#once a node has all children, can free register

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, input_nodes: List[ModelNode] = [], output_nodes: List[ModelNode] = list()) -> None:
		self._input_nodes: List[ModelNode] = input_nodes 
		self._output_nodes: List[ModelNode] = output_nodes 
	def to_torch_module_src(self, name: str) -> str:
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
		def format_component(component: int | Tuple[int, int]) -> str:
			if isinstance(component, int):
				return f"c{component}"
			else:
				return f"c{component[0]}_{component[1]}"
		def format_components(components: List[int | Tuple[int, int]]) -> List[str]:
			return [format_component(c) for c in components]
		def format_register(register: int) -> str:
			return f"r{register}"
		def format_registers(registers: List[int]) -> List[str]:
			return [format_register(r) for r in registers]
		for i, (node, register_out, registers_in) in enumerate(forward_data):
			forward_statment: str = node.get_schema_node().get_merge_method().get_merge_src(format_registers(registers_in))
			inits = node.get_schema_node().get_inits_src(node.get_mould_shape(), node.get_output_shape())
			if len(inits) > 0:
				components = [format_component((i, j)) for j in range(len(inits))]
				for component, init in zip(components, inits):
					init_statements.append(assign_(component, init))
				forward_statment = node.get_mould_view_src(forward_statment)
				for component in components:
					forward_statment = call_(component, forward_statment)
				forward_statment = node.get_output_view_src(forward_statment)
			forward_statements.append(assign_(format_register(register_out), forward_statment))
		src = pytorch_module_(name, init_statements, format_registers(list(range(len(self._input_nodes)))), forward_statements)
		return src 
	def get_model_handle(self, name: str) -> Type:
		return eval(self.to_torch_module_src(name))
		

class ModelNode():
	__slots__ = ["_index", "_id", "_schema_node", "_children", "_parents", "_output_shape", "_mould_shape"]
	def __init__(self, index: Index, id: int, node_pattern: SchemaNode, mould_shape: LockedShape, output_shape: LockedShape, parents: Iterable[Self] | None) -> None:
		self._index: Index = index
		self._id: int = id 
		self._schema_node: SchemaNode = node_pattern 
		self._children: List[Self] = []
		self._parents: List[Self] = []
		if parents is not None:
			self.set_parents(parents)
		self._mould_shape: LockedShape = mould_shape 
		self._output_shape: LockedShape = output_shape
	def add_child(self, child: Self) -> None:
		if child not in self._children:
			self._children.append(child)
			child.add_parent(self)
	def add_parent(self, parent: Self) -> None:
		if parent not in self._parents:
			self._parents.append(parent)
			parent.add_child(self)
	def set_parents(self, parents: Iterable[Self]) -> None:
		self._parents = [] 
		for parent in parents:
			self.add_parent(parent)
	def get_parents(self) -> List[Self]:
		return self._parents
	def get_children(self) -> List[Self]:
		return self._children
	def get_output_shape(self) -> LockedShape:
		return self._output_shape
	def get_mould_shape(self) -> LockedShape:
		return self._mould_shape
	def unbind(self) -> None:
		for child in self._children:
			child.unbind_parent(self)
		for parent in self._parents:
			parent.unbind_child(self)
	def unbind_child(self, child: Self) -> None:
		if child in self._children:
			self._children.remove(child)
			child.unbind_parent(self)
	def unbind_parent(self, parent: Self) -> None:
		if parent in self._parents:
			self._parents.remove(parent)
			parent.unbind_child(self)
	def get_schema_node(self) -> SchemaNode:
		return self._schema_node
	def get_dimensionality(self) -> int:
		return len(self._mould_shape) 
	def is_leaf(self) -> bool:
		return len(self._children) == 0
	def get_id(self) -> int:
		return self._id
	def get_components_src(self) -> Tuple[str | None, str | None, str | None]:
		#transform, activation, and batch norm
		return "", "", ""
	def get_output_view_src(self, tensor: str) -> str:
		return flatten_view_(tensor, self._output_shape)
	def get_mould_view_src(self, tensor: str) -> str:
		return view_(tensor, self._mould_shape)
