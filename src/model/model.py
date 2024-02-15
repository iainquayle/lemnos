from __future__ import annotations

import traceback

from src.schema.schema_node import SchemaNode
from src.shared.shape import LockedShape
from src.shared.index import Index

from typing import List, Tuple, Iterable, Dict, Set
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
	def to_torch_module_src(self, name: str | None = None) -> Tuple[str, str]:
		forward_info: List[Tuple[ModelNode, int, List[int]]] = []
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
						forward_info.append((node, register_out, registers_in))
		forward_src = ""
		init_src = ""
		for i, (node, register_out, registers_in) in enumerate(forward_info):
			transform_src, activation_src, regularization_src = node.get_components_src()
			if transform_src is not None:
				init_src += f"\t\tt{i} = {transform_src}\n"
			if activation_src is not None:
				init_src += f"\t\ta{i} = {activation_src}\n"
			if regularization_src is not None:
				init_src += f"\t\tb{i} = {regularization_src}\n"

			forward_src += f"\t\tr{register_out} = m{i}\n"
		src = "import torch\nimport torch.nn as nn\n" + \
			f"class {name if name is not None else 'Model'}(nn.Module):\n" + \
			"\tdef __init__(self):\n" + \
			"\t\tsuper().__init__()\n" + \
			f"{init_src}" + \
			f"\tdef forward(self, r{'r,'.join([str(r) for r in range(len(self._input_nodes))])})\n" + \
			f"{forward_src}" + \
			f"\t\treturn r{'r,'.join([str(r) for r in output_registers])}\n"
		return "",src 

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
	def unbind_children(self) -> None:#TODO: refactor unbinding
		self._children = []
	def unbind(self) -> None:
		if len(self._children) > 0:
			raise Exception("Cannot unbind node with children")
		for parent in self._parents:
			parent.unbind_child(self)
	def unbind_all(self) -> None:
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
	def is_leaf(self) -> bool:
		return len(self._children) == 0
	def get_id(self) -> int:
		return self._id
	def get_components_src(self) -> Tuple[str | None, str | None, str | None]:
		
		#transform, activation, and batch norm
		return "", "", ""
