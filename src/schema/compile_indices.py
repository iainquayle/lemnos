from __future__ import annotations

from ..shared import LockedShape, ID
from .schema_graph import SchemaNode
from .compile_index import CompileIndex 
from .ir_node import IRNode

from abc import ABC as Abstract, abstractmethod

import random

class CompilationIndices(Abstract):
	@abstractmethod
	def get_index(self, id: ID, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[CompileIndex, int]:	
		pass

class SequenceIndices(CompilationIndices):
	__slots__ = ["_indices"]
	def __init__(self, indices: dict[ID, CompileIndex]) -> None:
		self._indices: dict[ID, CompileIndex] = indices
	@staticmethod
	def from_ir(ir: list[IRNode]) -> SequenceIndices:
		indices: dict[ID, CompileIndex] = {}
		for node in ir:
			indices[node.id] = node.index
		return SequenceIndices(indices)
	def get_index(self, id: ID, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[CompileIndex, int]:
		return self._indices[id], 0 

class BreedIndices(CompilationIndices):
	__slots__ = ["_sequences", "_sequence_change_prod", "_mutate_prod"]
	def __init__(self, sequence_change_prod: float = 0, mutate_prod: float = 0, sequences: list[list[tuple[CompileIndex, SchemaNode, LockedShape]]] = []) -> None:
		if sequence_change_prod < 0 or sequence_change_prod > 1 or mutate_prod < 0 or mutate_prod > 1:
			raise ValueError("Invalid probabilities")
		self._sequences: list[list[tuple[CompileIndex, SchemaNode, LockedShape]]] = sequences
		self._sequence_change_prod: float = sequence_change_prod
		self._mutate_prod: float = mutate_prod
	@staticmethod
	def from_ir(ir: list[list[IRNode]], sequence_change_prod: float = 0, mutate_prod: float = 0) -> BreedIndices:
		pass
	def get_index(self, id: ID, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[CompileIndex, int]:
		def search_sequence(sequence_index: int) -> tuple[CompileIndex, int] | None:
			sequence_index %= len(self._sequences)
			min_diff: int = 2**32
			result: CompileIndex | None = None
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
				sequence_indices: list[int] = list(range(sequence_index)) + list(range(sequence_index + 1, len(self._sequences)))
				random.shuffle(sequence_indices)
				for sequence in sequence_indices:
					if (result := search_sequence(sequence)) is not None:
						index, _ = result
						return index, sequence 
		return CompileIndex.random(), sequence_index 
