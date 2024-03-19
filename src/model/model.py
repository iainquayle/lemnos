from __future__ import annotations

from ..shared import Index, LockedShape, OpenShape, Shape
from ..schema.schema import Schema, BuildIndices
from ..schema.schema_graph import SchemaNode, Transition, TransitionGroup, JoinType
from ..schema.src_generation import *

from typing import List, Tuple, Dict, Type

from copy import copy

class Model():
	_MAX_ITERATIONS = 1024 
	def __init__(self, schema: Schema, input_shapes: List[LockedShape], indices: BuildIndices, max_nodes: int = 512) -> None:
		self._input_nodes: List[ModelNode] = [] 
		self._output_nodes: List[ModelNode] = [] 
		self._ordered_node_cache: List[ModelNode] | None = []
		stacks: Dict[SchemaNode, _BuildStack] = {}
		for i, (schema_node, priority) in enumerate(schema.get_node_with_priority()):
			node = ModelNode(schema_node, input_shapes[i])
			self._input_nodes.append(node)
			stacks[schema_node] = _BuildStack(schema_node, [(node, priority)])
		tracker = _BuildTracker(max_nodes, stacks, {schema_node: 1 for schema_node in schema.get_starts_iter()})
		if ((first_node := tracker.pop_min()) is not None
				and (nodes := first_node.attempt_build(tracker, indices, 0, 0)) is not None):
			self._ordered_node_cache = nodes
			for end in schema.get_ends_iter():
				for node in nodes:
					if node.get_schema_node() == end:
						self._output_nodes.append(node)
		else:
			raise ValueError("Failed to build model")
	def get_ordered_nodes(self) -> List[ModelNode]:
		if self._ordered_node_cache is not None and len(self._ordered_node_cache) > 0:
			return self._ordered_node_cache
		else:
			evaluation_tracker: Dict[ModelNode, int] = {}
			ordered_nodes: List[ModelNode] = []
			for node in self._input_nodes:
				evaluation_tracker[node] = 0
			evaluated_node: bool = True
			while evaluated_node:
				evaluated_node = False
				for node, visits in list(evaluation_tracker.items()):
					if visits == len(node.get_parents()):
						del evaluation_tracker[node]
						ordered_nodes.append(node)
						evaluated_node = True
						for child in node.get_children():
							evaluation_tracker[child] = evaluation_tracker.get(child, 0) + 1
			self._ordered_node_cache = ordered_nodes.sort(key=lambda node: node.get_id())
			return copy(ordered_nodes) 
	def get_index_list(self) -> List[Index]:
		return [node._index for node in self.get_ordered_nodes()]
	def get_index_schema_list(self) -> List[Tuple[Index, SchemaNode]]:
		return [(node._index, node.get_schema_node()) for node in self.get_ordered_nodes()]
	def get_torch_module_src(self, name: str) -> str:
		forward_data: List[Tuple[ModelNode, int, List[int]]] = []
		output_registers: List[int] = []
		evaluation_tracker: Dict[ModelNode, List[int]] = {} #node and the registers it is using
		register_commitments: Dict[int, int] = {} #register and nodes it still needs to be used for
		available_registers: List[int] = []
		ordered_nodes: List[ModelNode] = self.get_ordered_nodes()
		register_count: int = 0
		i = 0
		for node in ordered_nodes:
			if node.is_root():
				pass
			else:
				pass
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
		def format_component(node: int, component: int | None = None) -> str:
			return f"c{node}" if component is None else f"c{node}_{component}"
		def format_register(register: int) -> str:
			return f"r{register}"
		def format_registers(registers: List[int]) -> List[str]:
			return [format_register(r) for r in registers]
		for i, (node, register_out, registers_in) in enumerate(forward_data):
			forward_statment: str = node.get_schema_node().get_merge_method().get_merge_src(format_registers(registers_in))
			inits = node.get_inits_src()
			if len(inits) > 0:
				components = [format_component(i, j) for j in range(len(inits))]
				for component, init in zip(components, inits):
					init_statements.append(self_(assign_(component, init)))
				forward_statment = node.get_mould_view_src(forward_statment)
				for component in components:
					forward_statment = self_(call_(component, forward_statment))
				forward_statment = node.get_output_view_src(forward_statment)
			if len(node.get_children()) == 0:
				forward_statment = node.get_final_view_shape(forward_statment) 
			forward_statements.append(assign_(format_register(register_out), forward_statment))
		forward_statements.append(return_(*format_registers(output_registers)))
		src = torch_module_(name, init_statements, format_registers(list(range(len(self._input_nodes)))), forward_statements)
		return src 
	def get_torch_module_handle(self, name: str) -> Type:
		exec(self.get_torch_module_src(name))
		return eval(name)

class ModelNode():
	_NOT_BUILT = -1
	__slots__ = ["_index", "_id", "_schema_node", "_children", "_parents", "_output_shape", "_mould_shape"]
	def __init__(self, 
			schema_node: SchemaNode,
			mould_shape: LockedShape = LockedShape(0),
			) -> None:
		self._index: Index = Index() 
		self._id: int = ModelNode._NOT_BUILT 
		self._schema_node: SchemaNode = schema_node 
		self._children: List[ModelNode] = []
		self._parents: List[ModelNode] = []
		self._mould_shape: LockedShape = mould_shape 
		self._output_shape: LockedShape = LockedShape(0)
	def attempt_build(self, build_tracker: _BuildTracker,  indices: BuildIndices, id: int, sequence_index: int) -> List[ModelNode] | None: 
		if self.is_built():
			raise ValueError("Cannot build node that is already built")
		if len(self._parents) != 0:
			self._mould_shape = self._schema_node.get_mould_shape([parent.get_output_shape() for parent in self._parents])
		self._id = id
		index, sequence_index = indices.get_index(id, sequence_index, self._schema_node, self._mould_shape)
		pivot = index.get_shuffled(len(self._schema_node.get_transition_groups()))
		i = 0
		while abs(i) <= max(len(self._schema_node.get_transition_groups()) - pivot, pivot):
			if pivot + i < len(self._schema_node.get_transition_groups()) and pivot + i >= 0:
				group = self._schema_node[pivot + i]
				next_tracker = copy(build_tracker)
				if ((nodes := next_tracker.record_and_get(group, self)) is not None
						and self.attempt_join_children(nodes, index)
					 	and (next_node := next_tracker.pop_min()) is not None #dont think this would happen?
						and (built_nodes := next_node.attempt_build(next_tracker, indices, id + 1, sequence_index)) is not None):
					return built_nodes + [self]
			i = -i if i > 0 else -i + 1
		if len(self._schema_node.get_transition_groups()) == 0:
			if (output_shape := self._schema_node.get_output_shape(self._mould_shape, OpenShape(), index)) is not None:
				self._output_shape = output_shape
				return [self]
			else:
				return None
		else:
			self._unbind()
			return None 
	def attempt_join_children(self, children: List[ModelNode], index: Index) -> bool:
		if ((conformance_shape := Shape.reduce_common_lossless([child.get_conformance_shape() for child in children])) is not None 
				and (output_shape := self._schema_node.get_output_shape(self._mould_shape, conformance_shape, index)) is not None):
			self._output_shape = output_shape
			self._set_children(children)
			return True
		return False
	def get_conformance_shape(self) -> Shape:
		return self._schema_node.get_conformance_shape([parent.get_output_shape() for parent in self._parents])
	def _unbind(self) -> None:
		self._unbind_children()
		self._unbind_parents()
	def _unbind_children(self) -> None:
		for child in self._children:
			child._parents.remove(self)
		self._children = []
	def _unbind_parents(self) -> None:
		for parent in self._parents:
			parent._children.remove(self)
		self._parents = []
	def _add_child(self, child: ModelNode) -> None: 
		if child not in self._children:
			self._children.append(child)
			child._add_parent(self)
	def _add_parent(self, parent: ModelNode) -> None:
		if parent not in self._parents:
			self._parents.append(parent)
			parent._add_child(self)
	def _set_parents(self, parents: Iterable[ModelNode]) -> None: #could find intersection of old and new parents to minimize unbinding
		self._unbind_parents()
		for parent in parents:
			self._add_parent(parent)
	def _set_children(self, children: Iterable[ModelNode]) -> None:
		self._unbind_children()
		for child in children:
			self._add_child(child)
	def get_children(self) -> List[ModelNode]:
		return copy(self._children)
	def get_parents(self) -> List[ModelNode]:
		return copy(self._parents)
	def is_built(self) -> bool:
		return self._id > ModelNode._NOT_BUILT
	def get_output_shape(self) -> LockedShape:
		if not self.is_built():
			raise ValueError("Cannot get output shape of unbuilt node")
		return self._output_shape
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
	def is_root(self) -> bool:
		return len(self._parents) == 0
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
	def __init__(self, max_nodes: int, stacks: Dict[SchemaNode, _BuildStack], node_counts: Dict[SchemaNode, int]) -> None:
		self._stacks: Dict[SchemaNode, _BuildStack] = stacks 
		self._node_counts: Dict[SchemaNode, int] = node_counts
		self._max_nodes: int = max_nodes
	def pop_min(self) -> ModelNode | None:
		min_priority = Transition.get_max_priority() + 1
		min_schema: SchemaNode | None = None
		for schema, stack in self.get_iter():
			if stack.get_priority() < min_priority:
				min_priority = stack.get_priority()
				min_schema = schema
		if min_schema is not None:
			return self._stacks[min_schema].pop()[_BuildStack.NODE]
		else:
			return None
	def record_and_get(self, transition_group: TransitionGroup, parent: ModelNode) -> List[ModelNode] | None:
		nodes: List[ModelNode] = []
		for transition in iter(transition_group):
			if transition.get_next() not in self:
				self._stacks[transition.get_next()] = _BuildStack(transition.get_next(), [])
			self._node_counts[transition.get_next()] = self._node_counts.get(transition.get_next(), 0) + 1 
			if (node := self._stacks[transition.get_next()].record_and_get(parent, transition.get_join_type(), transition.get_priority())) is not None:
				nodes.append(node)
			else:
				return None
		return nodes
	def is_empty(self) -> bool:
		for _, stack in self.get_iter():
			if len(stack) > 0:
				return False
		return True
	def stacks_str(self) -> str:
		return " , ".join([schema.debug_name + ": " + str(len(stack)) for schema, stack in self.get_iter()])
	def __copy__(self) -> _BuildTracker:
		return _BuildTracker(self._max_nodes, {key: copy(value) for key, value in self.get_iter()}, copy(self._node_counts))
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._stacks
	def __len__(self) -> int:
		return len(self._stacks)
	def get_iter(self) -> Iterable[Tuple[SchemaNode, _BuildStack]]:
		return iter(self._stacks.items())

class _BuildStack:
	NODE = 0
	PRIORITY = 1
	__slots__ = ["_stack", "_schema_node"]
	def __init__(self, schema_node: SchemaNode, stack: List[Tuple[ModelNode, int]]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: List[Tuple[ModelNode, int]] = stack
	def record_and_get(self, parent: ModelNode | SchemaNode, join_type: JoinType, priority: int) -> ModelNode | None: 
		if isinstance(parent, ModelNode):
			parent = parent.get_schema_node()
		if join_type != JoinType.NEW:
			for i, (node, _) in enumerate(self._stack):
				if not node.has_parent_type(parent):
					self._stack[i] = (node, priority if priority < self._stack[i][_BuildStack.PRIORITY] else self._stack[i][_BuildStack.PRIORITY])
					return node
		if join_type != JoinType.EXISTING:
			self._stack.append((ModelNode(self._schema_node), priority))
			return self.peek()[_BuildStack.NODE]
		else:
			return None
	def pop(self) -> Tuple[ModelNode, int]:
		return self._stack.pop()
	def peek(self) -> Tuple[ModelNode, int]:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek()[_BuildStack.PRIORITY] if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _BuildStack:
		return _BuildStack(self._schema_node, copy(self._stack))
