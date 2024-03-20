from __future__ import annotations

from ..shared import LockedShape
from .schema_graph import SchemaNode
from .compilation_indices import BuildIndices 
from .ir_compilation import IRNode, CompilationNodeStack, CompilationNode, CompilationTracker

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
	def compile_ir(self, input_shapes: list[LockedShape], build_indices: BuildIndices, max_nodes: int) -> list[IRNode] | None:
		tracker = CompilationTracker(
			[CompilationNodeStack(schema_node, [CompilationNode(set(), [], shape, i-len(input_shapes))]) for i, (schema_node, shape) in enumerate(zip(self._starts, input_shapes))], 
			None, 0, max_nodes)
		if (result := tracker.compile_ir(build_indices, 0)) is not None:
			return result.reverse()
		return None
