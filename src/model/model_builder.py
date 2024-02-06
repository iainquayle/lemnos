from __future__ import annotations

from src.model.model import ModelNode, Model
from src.schema.node_parameters import IdentityParameters 
from src.shared.shape import Bound, LockedShape, OpenShape
from src.shared.merge_method import Concat
from src.shared.index import Index
from src.schema.schema_node import SchemaNode, Transition

from typing import List, Dict, Tuple, Iterable

from copy import copy

#TODO: perhaps add a flag that switches whether the indices should be atModelNodeted in order, or just used at random for breeding,
#	could use a random seed, that can also effectivley work as a flag
#	only real option for capturing input and output nodes in current setup is to return a list of nodes and find them after
#TODO: consider adding ordering to input nodes to it 
#TODO: consider making turning join existing into enum
#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes
class ModelBuilder:
	def __init__(self, inputs: List[SchemaNode], outputs: List[SchemaNode], max_nodes: int) -> None:
		if len(inputs) == 0 or len(outputs) == 0:
			raise ValueError("No start or end patterns")
		self.inputs: List[SchemaNode] = inputs 
		self.outputs: List[SchemaNode] = outputs 
		self.max_nodes: int = max_nodes
	def add_start(self, pattern: SchemaNode) -> None:
		self.inputs.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self.outputs.append(pattern)
	def build(self, input_shapes: List[LockedShape], indices: List[Index]) -> Model:
		if len(input_shapes) != len(self.inputs):
			raise ValueError("Incorrect number of input shapes")
		return Model()

class _BuildTracker:
	__slots__ = ["_stacks", "_max_nodes", "_indices"]
	def __init__(self, indices: List[Index], max_nodes: int, stacks: Dict[SchemaNode, _BuildStack] = dict()) -> None:
		#self._stacks: List[Tuple[SchemaNode, _BuildStack]] = [(schema, stack) for schema, stack in stacks.items()] 
		self._stacks: Dict[SchemaNode, _BuildStack] = stacks 
		self._max_nodes: int = max_nodes
		self._indices: List[Index] = indices
	@staticmethod
	def build_nodes(inputs: Dict[SchemaNode, LockedShape], indices: List[Index], max_nodes: int) -> List[ModelNode] | None:
		dummy_nodes = {input_schema: ModelNode(Index(), -1, input_schema, shape, shape, None) for input_schema, shape in inputs.items()}
		tracker = _BuildTracker(indices, max_nodes, {input_schema: _BuildStack([_BuildNode([dummy_node], -1)]) for input_schema, dummy_node in dummy_nodes.items()})
		if isinstance((result := tracker._build_min(indices, 0)), List):
			for node in dummy_nodes.values():
				node.unbind_all()
			return result
		return None
	def _build_min(self, indices: List[Index], id: int) -> List[ModelNode] | SchemaNode:
		index = indices[0]
		if (result := self.pop_min()) is not None:
			schema_node, build_node = result
			parents = build_node.get_parents()
			input_shape: LockedShape = schema_node.get_merge_method().get_output_shape([parent.get_output_shape() for parent in parents])
			pivot = index.get_shuffled(len(schema_node.get_transition_groups()))
			i = 0
			while abs(i) <= max(len(schema_node.get_transition_groups()) - pivot, pivot):
				if pivot + i < len(schema_node.get_transition_groups()) and pivot + i >= 0:
					group = schema_node[pivot + i]
					transition_iter = iter(group)
					join_nodes: Dict[Transition, _BuildNode] = {}
					conformance_shape = OpenShape.new()
					while (transition := next(transition_iter, None)) is not None and conformance_shape is not None: #TODO: simplify somehow, fugly
						if transition.get_join_existing():
							if (join_on := self[transition.get_next()].get_available(schema_node)) is not None: 
								conformance_shape = conformance_shape.common_lossless(transition.get_next().get_conformance_shape(join_on.get_parent_shapes()))
								join_nodes[transition] = join_on
							else:
								conformance_shape = None
					if conformance_shape is not None:
						tracker_copy = copy(self)
						shapes = schema_node.get_parameters().get_mould_and_output_shapes(input_shape, conformance_shape, index)
						if shapes is not None:
							mould_shape, output_shape = shapes
							node = ModelNode(index, id, schema_node, mould_shape=mould_shape, output_shape=output_shape, parents=parents)
							transitions_recorded = True
							for transition in iter(group):
								transitions_recorded = transitions_recorded and tracker_copy.record_transition(transition, node)
							if id < self._max_nodes and transitions_recorded and isinstance(result := tracker_copy._build_min(indices, id + 1), List):
								return [node, *result]
								#two options:
								#	backtrack all the way to the creator of the node
								#		suppose node will always create shape out of bounds 
								#		quicker likely
								#		may miss some valid graphs?
								#	backtrack to the previous and try next option
								#		likely slower
								#		guaranteed to find all valid graphs constrained by known shortfalls
								#		definitely easier
								#would be benificial no matter which option, to do a preliminary bounds check on the transformed shape when the node is created
				i = -i if i > 0 else -i + 1
			if len(schema_node.get_transition_groups()) == 0:
				if (shapes := schema_node.get_parameters().get_mould_and_output_shapes(input_shape, OpenShape.new(), index)) is not None:
					return [ModelNode(index, id, schema_node, *shapes, parents)]
			return schema_node
		return []
	def min(self) -> Tuple[SchemaNode, _BuildStack] | None: 
		if len(self) == 0:
			return None
		min_schema = min(self.get_iter(), key=lambda item: item[1].get_priority()) 
		if len(min_schema[1]) == 0:
			return None
		return min_schema
	def pop_min(self) -> Tuple[SchemaNode, _BuildNode] | None:
		if (result := self.min()) is not None:
			schema, stack = result
			return schema, stack.pop()
		return None
	def record_transition(self, transition: Transition, parent: ModelNode) -> bool:
		if transition.get_join_existing():
			if transition.get_next() in self and (join_on_node := self[transition.get_next()].get_available(parent)) is not None:
				join_on_node.add_parent(parent, transition.get_priority())
				return True
			else:
				return False
		else:
			if transition.get_next() not in self:
				self[transition.get_next()] = _BuildStack([_BuildNode([parent], transition.get_priority())])
			else:
				self[transition.get_next()].push(_BuildNode([parent], transition.get_priority()))
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
		for schema, stack in self._stacks:
			if schema == key:
				return stack
		raise KeyError("Key not found")
	def __setitem__(self, key: SchemaNode, value: _BuildStack) -> None:
		self._stacks[key] = value
		return
		for i, (schema, _) in enumerate(self._stacks):
			if schema == key:
				self._stacks[i] = (schema, value)
				return
		self._stacks.append((key, value))
	def __copy__(self) -> _BuildTracker:
		return _BuildTracker(self._indices, self._max_nodes, {key: copy(value) for key, value in self.get_iter()})
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._stacks
		for schema, _ in self._stacks:
			if schema == key:
				return True
		return False
	def __len__(self) -> int:
		return len(self._stacks)
	def get_iter(self) -> Iterable[Tuple[SchemaNode, _BuildStack]]:
		return iter(self._stacks.items())
		return iter(self._stacks)
class _BuildNode:
	__slots__ = ["_parents", "_priority"]
	def __init__(self, parents: List[ModelNode], priority: int) -> None:
		self._parents: Dict[SchemaNode, ModelNode] = {parent.get_pattern(): parent for parent in parents} #may be quicker to make this a dict again
		self._priority: int = priority 
	def get_parent_shapes(self) -> List[LockedShape]:
		return [parent.get_output_shape() for parent in self._parents.values()]
	def get_parents(self) -> List[ModelNode]:
		return list(self._parents.values())
	def get_priority(self) -> int:
		return self._priority
	def add_parent(self, parent: ModelNode, priority: int) -> bool: #TODO: condider making this just return a bool
		if not self.available(parent):
			return False
		self._parents[parent.get_pattern()] = parent
		self._priority = min(self._priority, priority) 
		return True
	def available(self, parent: ModelNode | SchemaNode) -> bool:
		return (parent.get_pattern() if isinstance(parent, ModelNode) else parent) not in self._parents 
	def __copy__(self) -> _BuildNode:
		return _BuildNode(copy(self.get_parents()), self._priority)
class _BuildStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_BuildNode] = []) -> None:
		self._stack: List[_BuildNode] = stack 
	def push(self, data: _BuildNode) -> None:
		self._stack.append(data)
	def get_available(self, parent: ModelNode | SchemaNode) -> _BuildNode | None: 
		result = None
		for node in self._stack:
			if node.available(parent):
				result = node
		return result
	def pop(self) -> _BuildNode:
		return self._stack.pop()
	def peek(self) -> _BuildNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().get_priority() if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _BuildStack:
		return _BuildStack([copy(node) for node in self._stack])
