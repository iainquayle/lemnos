from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID

from abc import ABC as Abstract, abstractmethod

from random import random

def basic_or_search(schema: Schema, runner: Evaluator, max_id: ID, save_dir: str, model_pool_size: int = 1, breed_iterations: int = 1) -> None:
	test_indices: list[BreedIndices] = [BreedIndices() for _ in range(model_pool_size)]
	model_pool: list[tuple[list[IRNode], float]] = [] 
	failed_compilations = 0
	i = 0
	while i < breed_iterations: #will switch this to use a call back? allowing for an interactive cli?
		for j, indices in enumerate(test_indices):
			if (ir := schema.compile_ir(runner.get_input_shapes(), indices, max_id)) is not None:
				model_pool.append((ir, runner.evaluate_penalty(ir)))
			else:
				failed_compilations += 1
				if failed_compilations > 10:
					raise ValueError("Too many failed compilations")
			model_pool.sort(key=lambda pair: pair[1])
			model_pool = model_pool[:model_pool_size]
		test_indices = [BreedIndices([ir for ir, penalty in model_pool if random() < .2], .2, .2, .2) for _ in range(model_pool_size)]
		i += 1

class Evaluator(Abstract):
	@abstractmethod
	def evaluate_penalty(self, ir: list[IRNode]) -> float:
		pass
	@abstractmethod
	def get_input_shapes(self) -> list[LockedShape]:
		pass

class Tracker:
	def __init__(self, ir: list[IRNode], record_length: int) -> None:
		self._ir = ir
		self._record_length = record_length
		self._loss: list = [float]
