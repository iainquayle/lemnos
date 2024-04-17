from __future__ import annotations

from ..shared import LockedShape, ID
from .schema_graph import SchemaNode, IRNode, CompilationIndices, _CompilationTracker, _CompilationNode, _CompilationNodeStack

class Schema:
	def __init__(self, starts: list[SchemaNode], ends: list[SchemaNode]) -> None:
		if len(starts) == 0 or len(ends) == 0:
			raise ValueError("No start or end patterns")
		for end in ends:
			if len(end) > 0:
				raise ValueError("End patterns cannot not have transitions out")
		self._starts: list[SchemaNode] = starts 
		self._ends: list[SchemaNode] = ends 
	def add_start(self, pattern: SchemaNode) -> None:
		self._starts.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self._ends.append(pattern)
	def compile_ir(self, input_shapes: list[LockedShape], build_indices: CompilationIndices, max_id: ID) -> list[IRNode] | None:
		tracker = _CompilationTracker(
			[_CompilationNodeStack(schema_node, [_CompilationNode(set(), [], shape, i-len(input_shapes))]) for i, (schema_node, shape) in enumerate(zip(self._starts, input_shapes))], 
			None)
		schema, node = tracker.pop_min()
		ir = schema._compile(node, tracker, build_indices, ID(0), max_id)
		if ir is not None:
			ir.reverse()
			return ir 
		return None
