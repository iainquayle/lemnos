from __future__ import annotations

from ..shared import LockedShape, Shape, Index
from ..schema.schema_node import SchemaNode
from ..schema.src_generation import *

from typing import List, Iterable
from typing_extensions import Self

class ModelNode():
	__slots__ = ["_index", "_id", "_schema_node", "_children", "_parents", "_output_shape", "_mould_shape", "_mould_shape_invalid"]
	def __init__(self, 
			index: Index,
			id: int,
			schema_node: SchemaNode,
			mould_shape: LockedShape = LockedShape(0),
			output_shape: LockedShape = LockedShape(0),
			parents: Iterable[Self] | None = None
			) -> None:
		self._index: Index = index
		self._id: int = id 
		self._schema_node: SchemaNode = schema_node 
		self._children: List[Self] = []
		self._parents: List[Self] = []
		if parents is not None:
			self.set_parents(parents)
		self._mould_shape: LockedShape = mould_shape 
		self._output_shape: LockedShape = output_shape
		self._mould_shape_invalid: bool = False
	#perhaps rename to attepmt exapand, or attempt build
	def attempt_set_children(self, children: List[Self], index: Index) -> bool:
		if ((conformance_shape := Shape.reduce_common_lossless([child.get_output_shape() for child in children])) is not None 
				and (output_shape := self._schema_node.get_output_shape(self.get_set_mould_shape(), conformance_shape, index)) is not None):
			self._output_shape = output_shape
			self._set_children(children)
			return True
		return False
	def get_conformance_shape(self) -> Shape:
		return self._schema_node.get_conformance_shape([parent.get_output_shape() for parent in self._parents])
	def __del__(self) -> None:
		self.unbind()
	def unbind(self) -> None:
		self.unbind_children()
		self.unbind_parents()
	def unbind_children(self) -> None:
		for child in self._children:
			child._unbind_parent(self)
		self._children = []
	def unbind_parents(self) -> None:
		for parent in self._parents:
			parent._unbind_child(self)
		self._parents = []
	def _unbind_child(self, child: Self) -> None:
		self._children.remove(child)
	def _unbind_parent(self, parent: Self) -> None:
		self._parents.remove(parent)
	def add_child(self, child: Self) -> None:
		if child not in self._children:
			self._children.append(child)
			child._add_parent(self)
	def add_parent(self, parent: Self) -> None:
		if parent not in self._parents:
			self._mould_shape_invalid = True
			self._parents.append(parent)
			parent._add_child(self)
	def _add_child(self, child: Self) -> None:
		self._children.append(child)
	def _add_parent(self, parent: Self) -> None:
		self._mould_shape_invalid = True
		self._parents.append(parent)
	def set_parents(self, parents: Iterable[Self]) -> None: #technically could cause errors, perhaps make it private
		self.unbind_parents()
		for parent in parents:
			self.add_parent(parent)
	def _set_children(self, children: Iterable[Self]) -> None:
		self.unbind_children()
		for child in children:
			self.add_child(child)
	def get_output_shape(self) -> LockedShape:
		return self._output_shape
	def get_set_mould_shape(self) -> LockedShape: #merge these two later
		return self.get_mould_shape()
	def get_mould_shape(self) -> LockedShape:
		if self._mould_shape_invalid:
			self._mould_shape = self._schema_node.get_mould_shape([parent.get_output_shape() for parent in self._parents])
		return self._mould_shape
	def get_parents(self) -> List[Self]: #remove these later
		return self._parents
	def get_children(self) -> List[Self]:
		return self._children
	def get_schema_node(self) -> SchemaNode:
		return self._schema_node
	def dimensionality(self) -> int:
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
