from __future__ import annotations

from ..shared import LockedShape, ID 
from .schema_graph import SchemaNode, CompilationIndices, CompilationIndex, IRNode

import random
from copy import copy

import csv

class SequenceIndices(CompilationIndices):
	__slots__ = ["_indices"]
	def __init__(self, ir_or_load_path: list[IRNode] | str) -> None:
		if isinstance(ir_or_load_path, str):
			with open(ir_or_load_path, 'r') as file:
				self._indices = {ID(int(id)): CompilationIndex(int(index)) for id, index in csv.reader(file)}
		else:
			self._indices: dict[ID, CompilationIndex] = {node.id: node.index for node in ir_or_load_path} 
	def get_index(self, id: ID, schema_node: SchemaNode, shape_in: LockedShape) -> CompilationIndex:
		if id in self._indices:
			return self._indices[id]
		else:
			return CompilationIndex(0)
	def save(self, path: str) -> None:
		with open(path, 'w') as file:
			writer = csv.writer(file)
			for id, index in self._indices.items():
				writer.writerow([id, index])

class BreedIndices(CompilationIndices):
	__slots__ = ["_sequences", "_sequence_change_prob", "_ignore_shape_prob", "_mutate_prob", "_sequence_index", "_previous_id"]
	def __init__(self, ir_sequences: list[list[IRNode]] = [], sequence_change_prob: float = 0, ignore_shape_prob: float = 0, mutate_prob: float = 0) -> None:
		if sequence_change_prob < 0 or sequence_change_prob > 1 or mutate_prob < 0 or mutate_prob > 1:
			raise ValueError("Invalid probabilities")
		self._sequences: list[list[IRNode]] = [copy(sequence) for sequence in ir_sequences if len(sequence) != 0]
		for sequence in self._sequences:
			sequence.sort(key=lambda node: node.id)
		self._sequence_change_prob: float = sequence_change_prob
		self._ignore_shape_prob: float = ignore_shape_prob
		self._mutate_prob: float = mutate_prob
		self._sequence_index: int = 0
		self._previous_id: ID = ID(0)
	def get_index(self, id: ID, schema_node: SchemaNode, shape_in: LockedShape) -> CompilationIndex:
		def search_sequence(sequence_index: int, previous_id: ID) -> tuple[CompilationIndex, ID] | None:
			sequence_index %= len(self._sequences)
			min_diff: int = 2**32
			result: IRNode | None = None
			if random.random() < self._ignore_shape_prob:
				matching_nodes = [ir_node for ir_node in self._sequences[sequence_index]]
				return random.choice(matching_nodes).index, previous_id 
			for ir_node in self._sequences[sequence_index]:
				if (ir_node.schema_node == schema_node 
						and (diff := ir_node.input_shape.upper_difference(shape_in)) < min_diff 
						and ir_node.id > previous_id):
					min_diff = diff 
					result = ir_node 
			if result is not None:
				return result.index, result.id
			else:
				return None
		if random.random() < self._mutate_prob and len(self._sequences) != 0:
			if random.random() < self._sequence_change_prob or len(self._sequences) == 1:
				if (result := search_sequence(self._sequence_index, self._previous_id)) is not None:
					index, self._previous_id = result
					return index 
			if len(self._sequences) > 1:
				sequence_indices: list[int] = list(range(self._sequence_index)) + list(range(self._sequence_index + 1, len(self._sequences)))
				random.shuffle(sequence_indices)
				for sequence in sequence_indices:
					if (result := search_sequence(sequence, ID(0))) is not None:
						index, self._previous_id = result
						return index 
		return CompilationIndex.random() 

