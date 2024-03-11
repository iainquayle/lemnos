from __future__ import annotations

from ..shared import LockedShape, Shape, Index
from ..schema.schema_node import SchemaNode, JoinType
from ..schema.src_generation import *

from typing import List, Iterable, Dict, Tuple, Any
from typing_extensions import Self

from copy import copy

class ModelNode():
	_NOT_BUILT = -1
	__slots__ = ["_index", "_id", "_schema_node", "_children", "_parents", "_output_shape", "_mould_shape"]
	def __init__(self, 
			schema_node: SchemaNode,
			id: int = _NOT_BUILT, 
			index: Index = Index(),
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
			self._set_parents(parents)
		self._mould_shape: LockedShape = mould_shape 
		self._output_shape: LockedShape = output_shape
	def attempt_build(self, index: Index) -> bool: #will take in a new build tracker
		pass
	def attempt_join_children(self, children: List[Self], index: Index) -> bool:
		if ((conformance_shape := Shape.reduce_common_lossless([child.get_conformance_shape() for child in children])) is not None 
				and (output_shape := self._schema_node.get_output_shape(self._mould_shape, conformance_shape, index)) is not None):
			self._output_shape = output_shape
			self._set_children(children)
			return True
		return False
	def get_conformance_shape(self) -> Shape:
		return self._schema_node.get_conformance_shape([parent.get_output_shape() for parent in self._parents])
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
	def _add_child(self, child: Self) -> None: 
		if child not in self._children:
			self._children.append(child)
			child._add_parent(self)
	def _add_parent(self, parent: Self) -> None:
		if parent not in self._parents:
			self._parents.append(parent)
			parent._add_child(self)
	def _set_parents(self, parents: Iterable[Self]) -> None: #could find intersection of old and new parents to minimize unbinding
		self.unbind_parents()
		for parent in parents:
			self._add_parent(parent)
	def _set_children(self, children: Iterable[Self]) -> None:
		self.unbind_children()
		for child in children:
			self._add_child(child)
	def is_built(self) -> bool:
		return self._id > -1
	def get_output_shape(self) -> LockedShape:
		return self._output_shape
	def set_mould_shape(self) -> None:
		self._mould_shape = self._schema_node.get_mould_shape([parent.get_output_shape() for parent in self._parents])
	def get_mould_shape(self) -> LockedShape:
		if not self.is_built():
			raise ValueError("Cannot get mould shape of unbuilt node")
		return self._mould_shape
	def get_schema_node(self) -> SchemaNode:
		return self._schema_node
	def dimensionality(self) -> int:
		return self._schema_node.dimensionality()
	def is_leaf(self) -> bool:
		return len(self._children) == 0
	def has_parent_type(self, schema_node: SchemaNode) -> bool:
		for parent in self._parents:
			if parent.get_schema_node() == schema_node:
				return True
		return False
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

class _BuildTracker:
	__slots__ = ["_stacks", "_max_nodes", "_indices", "_node_counts", "_sequence_index"]
	def __init__(self, max_nodes: int, stacks: Dict[SchemaNode, _BuildStack], sequence_index: int) -> None:
		self._stacks: Dict[SchemaNode, _BuildStack] = stacks 
		self._node_counts: Dict[SchemaNode, int] = {}
		self._max_nodes: int = max_nodes
		self._sequence_index: int = sequence_index 
	def _build_min(self, indices: Any, depth: int) -> List[ModelNode] | SchemaNode:
		if (result := self._pop_min_node()) is not None:
			schema_node, build_node = result
			parents = build_node.get_parents()
			mould_shape = schema_node.get_mould_shape([parent.get_output_shape() for parent in parents])
			index, self._sequence_index = indices.get_index(depth, self._sequence_index, schema_node, mould_shape)
			pivot = index.get_shuffled(len(schema_node.get_transition_groups()))
			i = 0
			while abs(i) <= max(len(schema_node.get_transition_groups()) - pivot, pivot):
				if pivot + i < len(schema_node.get_transition_groups()) and pivot + i >= 0:
					group = schema_node[pivot + i]
					#will need to make function to get possible children of a group
					#	does anything need to be done for newly created nodes?
					#	they should auto delete if not used since everything unbinds
					#	the group get and record can be done in the same function
					#	needs to be done on the tracker copy though
					if ((conformance_shape := self._get_group_conformance_shape(group, schema_node)) is not None
			 				and (output_shape := schema_node.get_output_shape(mould_shape, conformance_shape, index)) is not None):
						next_tracker = copy(self)
						node = ModelNode(index, depth, schema_node, mould_shape, output_shape, parents)
						next_tracker._increment_count(schema_node)
						if (depth < self._max_nodes 
								and next_tracker._record_transitions(iter(group), node) 
								and isinstance(result := next_tracker._build_min(indices, depth + 1), List)):
							return [node, *result]
						else:
							node.unbind()	
				i = -i if i > 0 else -i + 1
			if len(schema_node.get_transition_groups()) == 0:
				if (output_shape := schema_node.get_output_shape(mould_shape, OpenShape(), index)) is not None:
					return [ModelNode(index, depth, schema_node, mould_shape, output_shape, parents)]
			return schema_node
		return []
	def _increment_count(self, schema_node: SchemaNode) -> None:
		self._node_counts[schema_node] = self._node_counts.get(schema_node, 0) + 1
	def _get_count(self, schema_node: SchemaNode) -> int:
		return self._node_counts.get(schema_node, 0)
	def _get_group_conformance_shape(self, group: TransitionGroup, schema_node: SchemaNode) -> Shape | None:
		transition_iter = iter(group)
		conformance_shape = OpenShape()
		while (transition := next(transition_iter, None)) is not None and conformance_shape is not None:
			if transition.is_join_existing():
				if (join_node := self[transition.get_next()].get_available(schema_node)) is not None: 
					conformance_shape = conformance_shape.common_lossless(transition.get_next().get_conformance_shape(join_node.get_parent_shapes()))
				else:
					conformance_shape = None
		return conformance_shape
	def _min_stack(self) -> Tuple[SchemaNode, _BuildStack] | None: 
		if len(self) == 0:
			return None
		min_schema = min(self.get_iter(), key=lambda item: item[1].get_priority()) 
		if len(min_schema[1]) == 0:
			return None
		return min_schema
	def _pop_min_node(self) -> Tuple[SchemaNode, ModelNode] | None:
		if (result := self._min_stack()) is not None:
			schema, stack = result
			return schema, stack.pop()
		return None
	def _record_transitions(self, transitions: Iterable[Transition], parent: ModelNode) -> bool:
		for transition in transitions:
			if not self.record_transition(transition, parent):
				return False
		return True
	def record_transition(self, transition: Transition, parent: ModelNode) -> bool:
		if transition.is_join_existing():
			if transition.get_next() in self and (join_on_node := self[transition.get_next()].get_available(parent)) is not None:
				join_on_node.add_parent(parent, transition.get_priority())
				return True
			else:
				return False
		else:
			if transition.get_next() not in self:
				self[transition.get_next()] = _BuildStack([ModelNode([parent], transition.get_priority())])
			else:
				self[transition.get_next()].push(ModelNode([parent], transition.get_priority()))
			return True	
	def is_empty(self) -> bool:
		for _, stack in self.get_iter():
			if len(stack) > 0:
				return False
		return True
	def stacks_str(self) -> str:
		return " , ".join([schema.debug_name + ": " + str(len(stack)) for schema, stack in self.get_iter()])
	def __getitem__(self, key: SchemaNode) -> _BuildStack:
		return self._stacks[key]
	def __setitem__(self, key: SchemaNode, value: _BuildStack) -> None:
		self._stacks[key] = value
		return
	def __copy__(self) -> _BuildTracker:
		return _BuildTracker(self._max_nodes, {key: copy(value) for key, value in self.get_iter()}, self._sequence_index)
	def _next_tracker(self) -> _BuildTracker:
		return _BuildTracker(self._max_nodes, {key: copy(value) for key, value in self.get_iter()}, self._sequence_index)
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._stacks
	def __len__(self) -> int:
		return len(self._stacks)
	def get_iter(self) -> Iterable[Tuple[SchemaNode, _BuildStack]]:
		return iter(self._stacks.items())

class _BuildStack:
	_NODE = 0
	_PRIORITY = 1
	__slots__ = ["_stack", "_schema"]
	def __init__(self, schema: SchemaNode, stack: List[Tuple[ModelNode, int]] = []) -> None:
		self._schema: SchemaNode = schema
		self._stack: List[Tuple[ModelNode, int]] = stack 
	def push(self, node: ModelNode, priority: int) -> None:
		self._stack.append((node, priority))
	def record_and_get(self, parent: ModelNode | SchemaNode, join_type: JoinType, priority: int) -> ModelNode | None: 
		if join_type != JoinType.NEW:
			for i, (node, _) in enumerate(self._stack):
				if not node.has_parent_type(parent.get_schema_node() if isinstance(parent, ModelNode) else parent):
					self._stack[i] = (node, priority)
					return node
		if join_type != JoinType.EXISTING:
			self.push(ModelNode(self._schema), priority)
		else:
			return None
	def pop(self) -> Tuple[ModelNode, int]:
		return self._stack.pop()
	def peek(self) -> Tuple[ModelNode, int]:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek()[_BuildStack._PRIORITY] if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _BuildStack:
		return _BuildStack(self._schema, copy(self._stack))
