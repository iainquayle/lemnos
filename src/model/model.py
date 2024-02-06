from __future__ import annotations

from src.schema.schema_node import SchemaNode
from src.shared.shape import LockedShape
from src.shared.index import Index

from typing import List, Tuple, Iterable
from typing_extensions import Self

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, input_nodes: List[ModelNode] = [], output_nodes: List[ModelNode] = list()) -> None:
		self._input_nodes: List[ModelNode] = input_nodes 
		self._output_nodes: List[ModelNode] = output_nodes 
	def to_flat_source_module(self) -> Tuple[str, str]:
		return "", ""

class ModelNode():
	__slots__ = ["_index", "_id", "_node_pattern", "_children", "_parents", "_output_shape", "_mould_shape"]
	def __init__(self, index: Index, id: int, node_pattern: SchemaNode, mould_shape: LockedShape, output_shape: LockedShape, parents: Iterable[Self] | None) -> None:
		self._index: Index = index
		self._id: int = id 
		self._node_pattern: SchemaNode = node_pattern 
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
	def get_output_shape(self) -> LockedShape:
		return self._output_shape
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
	def get_pattern(self) -> SchemaNode:
		return self._node_pattern

