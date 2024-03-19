from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, Index 
from .schema_graph import SchemaNode, TransitionGroup, Transition, JoinType

from typing import List, Tuple, Iterable, Set, Dict

import random
from copy import copy

from abc import ABC as Abstract, abstractmethod
from dataclasses import dataclass


class Schema:
	def __init__(self, starts: List[SchemaNode], ends: List[SchemaNode], max_nodes: int = 1024) -> None:
		if len(starts) == 0 or len(ends) == 0:
			raise ValueError("No start or end patterns")
		for end in ends:
			if len(end.get_transition_groups()) > 0:
				raise ValueError("End patterns cannot not have transitions out")
		self._starts: List[SchemaNode] = starts 
		self._ends: List[SchemaNode] = ends 
		self._max_nodes: int = max_nodes
	def add_start(self, pattern: SchemaNode) -> None:
		self._starts.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self._ends.append(pattern)
	def get_starts_iter(self) -> Iterable[SchemaNode]:
		return iter(self._starts)
	def get_ends_iter(self) -> Iterable[SchemaNode]:
		return iter(self._ends)
	def get_node_with_priority(self) -> List[Tuple[SchemaNode, int]]:
		return [(node, i - len(self._starts)) for i, node in enumerate(self._starts)]
	def compile_IR(self, input_shapes: List[LockedShape], build_indices: NodeCompileIndices, max_nodes: int) -> Tuple[str, StaticIndices]:
		raise NotImplementedError("Direct compile not implemented")


