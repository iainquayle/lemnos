from __future__ import annotations

from ..shared import LockedShape, ID
from .schema_graph import SchemaNode
from .compile_index import CompileIndex

from dataclasses import dataclass

@dataclass(frozen=True)
class IRNode:
	schema_node: SchemaNode
	parent_ids: tuple[ID, ...]
	id: ID 
	input_shape: LockedShape
	output_shape: LockedShape
	index: CompileIndex
	def __str__(self) -> str:
		return f"SchemaNode: {self.schema_node.debug_name}, Parent IDs: {self.parent_ids}, ID: {self.id}, Input Shape: {self.input_shape}, Output Shape: {self.output_shape}, CompileIndex: {self.index}"
