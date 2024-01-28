from __future__ import annotations

from src.schema.node_parameters import IdentityParameters
from src.schema.priority_graphs.manual import SchemaNode, Schema, Transition 
from src.shared.shape import LockedShape, OpenShape, Bound 
from src.shared.index import Index
from src.shared.merge_method import Concat

from typing import List, Set, Tuple, Iterable, Dict
from typing_extensions import Self

from dataclasses import dataclass
from copy import copy, deepcopy

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, input_nodes: List[ModelNode] = [], output_nodes: List[ModelNode] = list()) -> None:
		self._input_nodes: List[ModelNode] = input_nodes 
		self._output_nodes: List[ModelNode] = output_nodes 
	def to_flat_source_module(self) -> Tuple[str, str]:
		return "", ""
	@staticmethod
	def from_schema(schema: Schema, input_shapes: List[LockedShape], indices: List[Index]) -> Model:
		if len(input_shapes) != len(schema.inputs):
			raise ValueError("Incorrect number of input shapes")
		expansion_collection: _ExpansionCollection = _ExpansionCollection()
		input_nodes: List[ModelNode] = []
		for i, shape in enumerate(input_shapes):
			input = SchemaNode(IdentityParameters(Bound([None] * len(shape))), Concat())
			input_node = ModelNode(Index(), i, input, shape, shape, None)
			input_nodes.append(input_node)
			expansion_collection.add(input_node, -1)
		nodes = Model._generate_node_from_schema(expansion_collection, indices, len(input_nodes))
		if (result := expansion_collection.min()) is not None:
			to_expand, _ = result
			#to_expand.build(expansion_collection, indices, len(input_nodes))
		else:
			raise ValueError("No valid input")
		return Model()
	@staticmethod
	def _generate_node_from_schema(expansion_collection: _ExpansionCollection, indices: List[Index], id: int) -> List[ModelNode] | SchemaNode:
		index = indices[0]
		if (result := expansion_collection.min()) is not None:
			schema_node, stack = result
			parents = stack.pop().parents
			pivot = index.to_int(len(schema_node.transition_groups))
			i = 0
			while abs(i) < max(len(schema_node.transition_groups) - pivot, pivot):
				if pivot + i < len(schema_node.transition_groups) and pivot + i >= 0:
					group = schema_node.transition_groups[pivot + i]
					transition_iter = iter(group.transitions)
					join_nodes: Dict[Transition, _ExpansionNode] = {}
					conformance_shape = OpenShape.new()
					while (transition := next(transition_iter, None)) is not None and conformance_shape is not None:
						if transition.join_existing:
							if join_on := expansion_collection[transition.next].get_available(parents[0]):
								conformance_shape = conformance_shape.common_lossless(transition.next.get_conformance_shape(join_on.get_parent_shapes()))
								join_nodes[transition] = join_on
							else:
								conformance_shape = None
					if conformance_shape is not None:
						new_collection = copy(expansion_collection)
						shapes = schema_node.node_parameters.get_mould_and_output_shapes(parents[0].get_output_shape(), conformance_shape, index)
						if shapes is not None:
							node = ModelNode(index, id, schema_node, *shapes, parents)
							for transition in group.transitions:
								stack = new_collection[transition.next]
								if transition.join_existing:
									join_nodes[transition].add_parent(node, transition.priority)
								else:
									new_collection.add(node, transition.priority)
							if isinstance(result := Model._generate_node_from_schema(new_collection, indices, id + 1), SchemaNode):
								for transition in group.transitions:
									if transition.next == result and not transition.join_existing:
										return result
							else:
								return [node, *result]
				i = -i if i > 0 else -i + 1
			return schema_node
		return []

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

@dataclass
class _ExpansionNode:
	def __init__(self, parents: List[ModelNode], priority: int) -> None:
		self.parents: List[ModelNode] = parents #may be quicker to make this a dict again
		self.priority: int = priority 
	def get_parent_shapes(self) -> List[LockedShape]:
		return [parent.get_output_shape() for parent in self.parents]
	def add_parent(self, parent: ModelNode, priority: int) -> None: #TODO: condider making this just return a bool
		if not self.available(parent):
			raise ValueError("Parent already taken")
		self.parents.append(parent)
		self.priority = min(self.priority, priority) 
	def available(self, node: ModelNode) -> bool:
		for parent in self.parents:
			if parent.get_pattern() == node.get_pattern():
				return False 
		return True 
	def __copy__(self) -> _ExpansionNode:
		return _ExpansionNode(copy(self.parents), self.priority)
class _ExpansionStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_ExpansionNode] = []) -> None:
		self._stack: List[_ExpansionNode] = stack 
	def push(self, data: _ExpansionNode) -> None:
		self._stack.append(data)
	def get_available(self, node: ModelNode) -> _ExpansionNode | None: 
		for i in range(len(self._stack) - 1, -1, -1):
			if self._stack[i].available(node):
				return self._stack[i] 
		return None
	def pop(self) -> _ExpansionNode:
		return self._stack.pop()
	def peek(self) -> _ExpansionNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __deepcopy__(self) -> _ExpansionStack:
		return _ExpansionStack(copy(self._stack))
class _ExpansionCollection:
	__slots__ = ["_expansion_nodes"]
	def __init__(self, expansion_nodes: Dict[SchemaNode, _ExpansionStack] = dict()) -> None:
		self._expansion_nodes: Dict[SchemaNode, _ExpansionStack] = expansion_nodes
	def min(self) -> Tuple[SchemaNode, _ExpansionStack] | None:
		if len(self._expansion_nodes) == 0:
			return None
		min_schema = min(self._expansion_nodes.items(), key=lambda item: item[1].get_priority()) 
		if len(min_schema[1]) == 0:
			return None
		return min_schema
	def add(self, node: ModelNode, priority: int) -> None:
		if node.get_pattern() in self._expansion_nodes:
			self._expansion_nodes[node.get_pattern()].push(_ExpansionNode([node], priority))
		else:
			self._expansion_nodes[node.get_pattern()] = _ExpansionStack([_ExpansionNode([node], priority)])
	def __getitem__(self, key: SchemaNode) -> _ExpansionStack:
		return self._expansion_nodes[key]
	def __copy__(self) -> _ExpansionCollection:
		return _ExpansionCollection(deepcopy(self._expansion_nodes))
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._expansion_nodes

