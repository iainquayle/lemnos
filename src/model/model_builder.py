from __future__ import annotations

from src.model.model import ModelNode, Model
from src.schema.node_parameters import IdentityParameters 
from src.shared.shape import Bound, LockedShape, OpenShape
from src.shared.merge_method import Concat
from src.shared.index import Index
from src.schema.schema_node import SchemaNode, Transition 

from typing import List, Dict, Tuple
from copy import copy, deepcopy 

from dataclasses import dataclass

#TODO: perhaps add a flag that switches whether the indices should be atModelNodeted in order, or just used at random for breeding,
#	could use a random seed, that can also effectivley work as a flag
#	only real option for capturing input and output nodes in current setup is to return a list of nodes and find them after
#TODO: consider adding ordering to input nodes to it 
#TODO: consider making turning join existing into enum
#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes
class ModelBuilder:
	def __init__(self) -> None:
		if len(self.inputs) == 0 or len(self.outputs) == 0:
			raise ValueError("No start or end patterns")
		self.inputs: List[SchemaNode] = []
		self.outputs: List[SchemaNode] = []
	def add_start(self, pattern: SchemaNode) -> None:
		self.inputs.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self.outputs.append(pattern)
	def from_schema(self, input_shapes: List[LockedShape], indices: List[Index]) -> Model:
		if len(input_shapes) != len(self.inputs):
			raise ValueError("Incorrect number of input shapes")
		expansion_collection: _ExpansionCollection = _ExpansionCollection()
		input_nodes: List[ModelNode] = []
		for i, shape in enumerate(input_shapes):
			input = SchemaNode(IdentityParameters(Bound([None] * len(shape))), Concat())
			input_node = ModelNode(Index(), i, input, shape, shape, None)
			input_nodes.append(input_node)
			#expansion_collection.add(input_node, -1)
		nodes = ModelBuilder._build_node(expansion_collection, indices, len(input_nodes))
		if (result := expansion_collection.min()) is not None:
			to_expand, _ = result
			#to_expand.build(expansion_collection, indices, len(input_nodes))
		else:
			raise ValueError("No valid input")
		return Model()
	@staticmethod #TODO: it may make sense that this actually be put under the expansion collection class, as its expanding itself
	def _build_node(expansion_collection: _ExpansionCollection, indices: List[Index], id: int) -> List[ModelNode] | SchemaNode: 
		index = indices[0]
		if (result := expansion_collection.min()) is not None:
			schema_node, stack = result
			parents = stack.pop().parents
			pivot = index.get_shuffled(len(schema_node))
			i = 0
			while abs(i) < max(len(schema_node) - pivot, pivot):
				if pivot + i < len(schema_node) and pivot + i >= 0:
					group = schema_node[pivot + i]
					transition_iter = iter(group)
					join_nodes: Dict[Transition, _ExpansionNode] = {}
					conformance_shape = OpenShape.new()
					while (transition := next(transition_iter, None)) is not None and conformance_shape is not None:
						if transition.get_join_existing():
							if join_on := expansion_collection[transition.get_next()].get_available(parents[0]): #TODO: double check this
								conformance_shape = conformance_shape.common_lossless(transition.get_next().get_conformance_shape(join_on.get_parent_shapes()))
								join_nodes[transition] = join_on
							else:
								conformance_shape = None
					if conformance_shape is not None:
						new_collection = copy(expansion_collection)
						shapes = schema_node.get_parameters().get_mould_and_output_shapes(parents[0].get_output_shape(), conformance_shape, index)
						if shapes is not None:
							node = ModelNode(index, id, schema_node, *shapes, parents)
							for transition in iter(group):
								stack = new_collection[transition.get_next()]
								new_collection.add(transition, node)
							if isinstance(result := ModelBuilder._build_node(new_collection, indices, id + 1), SchemaNode):
								for transition in iter(group):
									if transition.get_next() == result and not transition.join_existing():
										return result
							else:
								return [node, *result]
				i = -i if i > 0 else -i + 1
			return schema_node
		return []

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
	def copy(self) -> _ExpansionNode:
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
	def copy(self) -> _ExpansionStack:
		return _ExpansionStack([copy(node) for node in self._stack])
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
	def pop_min(self) -> _ExpansionNode | None:
		if (result := self.min()) is not None:
			_, stack = result
			return stack.pop()
		return None
	def add(self, transition: Transition, parent: ModelNode) -> bool:
		if transition.get_join_existing():
			if transition.get_next() in self and (join_on := self[transition.get_next()].get_available(parent)) is not None:
				join_on.add_parent(parent, transition.get_priority())
				return True
			else:
				return False
		else:
			if transition.get_next() not in self:
				self._expansion_nodes[transition.get_next()] = _ExpansionStack()
			self._expansion_nodes[transition.get_next()].push(_ExpansionNode([parent], transition.get_priority()))
			return True	
	def __getitem__(self, key: SchemaNode) -> _ExpansionStack:
		return self._expansion_nodes[key]
	def __copy__(self) -> _ExpansionCollection:
		return _ExpansionCollection({key: copy(value) for key, value in self._expansion_nodes.items()})
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._expansion_nodes
	def __len__(self) -> int:
		return len(self._expansion_nodes)

