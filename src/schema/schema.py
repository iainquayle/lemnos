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


class BuildIndices(Abstract):
	@abstractmethod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:	
		pass

class StaticIndices(BuildIndices):
	__slots__ = ["_indices"]
	def __init__(self, indices: List[Index]) -> None:
		self._indices: List[Index] = indices
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:
		return self._indices[id], 0 

class BreedIndices(BuildIndices):
	__slots__ = ["_sequences", "_sequence_change_prod", "_mutate_prod"]
	def __init__(self, sequence_change_prod: float = 0, mutate_prod: float = 0, sequences: List[List[Tuple[Index, SchemaNode, LockedShape]]] = []) -> None:
		if sequence_change_prod < 0 or sequence_change_prod > 1 or mutate_prod < 0 or mutate_prod > 1:
			raise ValueError("Invalid probabilities")
		self._sequences: List[List[Tuple[Index, SchemaNode, LockedShape]]] = sequences
		self._sequence_change_prod: float = sequence_change_prod
		self._mutate_prod: float = mutate_prod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:
		def search_sequence(sequence_index: int) -> Tuple[Index, int] | None:
			sequence_index %= len(self._sequences)
			min_diff: int = 2**32
			result: Index | None = None
			for index, node, shape in self._sequences[sequence_index]:
				if node == schema_node and (diff := shape.upper_difference(shape_in)) < min_diff:
					min_diff = diff 
					result = index 
			if result is not None:
				return result, min_diff 
			else:
				return None
		if random.random() > self._mutate_prod and len(self._sequences) != 0:
			if random.random() > self._sequence_change_prod or len(self._sequences) == 1:
				if (result := search_sequence(sequence_index)) is not None:
					index, _ = result
					return index, sequence_index
			if len(self._sequences) > 1:
				sequence_indices: List[int] = list(range(sequence_index)) + list(range(sequence_index + 1, len(self._sequences)))
				random.shuffle(sequence_indices)
				for sequence in sequence_indices:
					if (result := search_sequence(sequence)) is not None:
						index, _ = result
						return index, sequence 
		return Index.random(), sequence_index 

