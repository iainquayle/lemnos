from __future__ import annotations

from src.shared import LockedShape, Index
from src.schema.schema_node import SchemaNode
from src.schema.src_generation import *

from typing import List, Iterable
from typing_extensions import Self

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
			child._unbind_parent(self)
		for parent in self._parents:
			parent._unbind_child(self)
		self._children = []
		self._parents = [] 
	def _unbind_child(self, child: Self) -> None:
		self._children.remove(child)
	def _unbind_parent(self, parent: Self) -> None:
		self._parents.remove(parent)
	def get_schema_node(self) -> SchemaNode:
		return self._schema_node
	def get_dimensionality(self) -> int:
		return len(self._mould_shape) 
	def is_leaf(self) -> bool:
		return len(self._children) == 0
	def get_id(self) -> int:
		return self._id
	def get_inits_src(self) -> List[str]:
		return self._schema_node.get_inits_src(self._mould_shape, self._output_shape)
	def get_output_view_src(self, tensor: str) -> str:
		return flatten_view_(tensor, self._output_shape)
	def get_mould_view_src(self, tensor: str) -> str:
		return view_(tensor, self._mould_shape)
	def get_final_view_shape(self, tensor: str) -> str:
		return view_(tensor, self._output_shape)
