from __future__ import annotations

from src.schema.priority_graphs.manual import SchemaNode 
from src.shared.shape import LockedShape 
from src.shared.index import Index

from typing import List, Set, Tuple, Iterable
from typing_extensions import Self

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, input_nodes: List[ModelNode] = [], output_nodes: List[ModelNode] = list()) -> None:
		self._input_nodes: List[ModelNode] = input_nodes 
		self._output_nodes: List[ModelNode] = output_nodes 
	def to_flat_source_module(self) -> Tuple[str, str]:
		return "", ""
	def to_runnable_module(self) -> None:
		pass

class ModelNode():
	__slots__ = ["_index", "_id", "_node_pattern", "_children", "_parents", "_output_shape", "_mould_shape"]
	def __init__(self, index: Index, id: int, node_pattern: SchemaNode, output_shape: LockedShape, mould_shape: LockedShape, parents: Iterable[Self] | None) -> None:
		self._index: Index = index
		self._id: int = id 
		self._node_pattern: SchemaNode = node_pattern 
		self._children: Set[Self] = set() #may want to make this a list, so that order is preserved
		self._parents: Set[Self] = set()
		if parents is not None:
			self.set_parents(parents)
		self._output_shape: LockedShape = output_shape
		self._mould_shape: LockedShape = mould_shape 
	def add_child(self, child: Self) -> None:
		if child not in self._children:
			self._children.add(child)
			child.add_parent(self)
	def add_parent(self, parent: Self) -> None:
		if parent not in self._parents:
			self._parents.add(parent)
			parent.add_child(self)
	def set_parents(self, parents: Iterable[Self]) -> None:
		self._parents = set()
		for parent in parents:
			self.add_parent(parent)
	def get_output_shape(self) -> LockedShape:
		return self._output_shape
	def unbind(self) -> None:
		if len(self._children) > 0:
			raise Exception("Cannot unbind node with children")
		for parent in self._parents:
			parent.unbind_child(self)
	def unbind_child(self, child: Self) -> None:
		if child in self._children:
			self._children.remove(child)
			child.unbind_parent(self)
	#technically dont need to unbind parent, but probably safest
	def unbind_parent(self, parent: Self) -> None:
		if parent in self._parents:
			self._parents.remove(parent)
			parent.unbind_child(self)
	def get_pattern(self) -> SchemaNode:
		return self._node_pattern
