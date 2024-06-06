from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID

from abc import ABC as Abstract, abstractmethod

from copy import copy

#ModelPool = list[tuple[list[IRNode], Metrics, Metrics | None]]
def or_search(schema: Schema, evaluator: Evaluator, selector: Selector, max_id: ID, save_dir: str, model_pool_size: int = 1, breed_iterations: int = 1) -> None:
	test_indices: list[BreedIndices] = [BreedIndices() for _ in range(model_pool_size)]
	model_pool: list[tuple[list[IRNode], Metrics, Metrics | None]] = [] 
	failed_compilations = 0
	i = 0
	while i < breed_iterations: #will switch this to use a call back? allowing for an interactive cli?
		for j, indices in enumerate(test_indices):
			if (ir := schema.compile_ir(evaluator.get_input_shapes(), indices, max_id)) is not None:
				training_metrics, validation_metrics = evaluator.evaluate(ir)
				model_pool.append((ir, training_metrics, validation_metrics))
			else:
				failed_compilations += 1
				if failed_compilations > 10:
					raise ValueError("Too many failed compilations")
		test_indices = selector.select(model_pool)
		i += 1

class Selector(Abstract):
	@abstractmethod
	def select(self, models: list[tuple[list[IRNode], Metrics, Metrics | None]]) -> list[BreedIndices]:
		pass

class MinAvgLossSelector(Selector):
	def select(self, models: list[tuple[list[IRNode], Metrics, Metrics | None]]) -> list[BreedIndices]:
		#models.sort(key=lambda pair: pair[1].avg_loss)
		return [BreedIndices([ir for ir, _, _ in models[:len(models) // 2]], .2, .2, .2) for _ in models]

class Evaluator(Abstract):
	@abstractmethod
	def evaluate(self, ir: list[IRNode]) -> tuple[Metrics, Metrics | None]:
		pass
	@abstractmethod
	def get_input_shapes(self) -> list[LockedShape]:
		pass

class SampleCollection:
	__slots__ = ["sample_size", "avg_loss", "max_loss", "min_loss", "accuracy", "time", "epoch"]
	def __init__(self, avg_loss: float, max_loss: float, min_loss: float, accuracy: float | None, time: float | None, epoch: int | None, sample_size: int = 1) -> None:
		self.sample_size: int = sample_size 
		self.avg_loss: float = avg_loss
		self.max_loss: float = max_loss
		self.min_loss: float = min_loss 
		self.accuracy: float | None = accuracy 
		self.time: float | None = time 
		self.epoch: int | None = epoch
	def merge(self, other: SampleCollection) -> SampleCollection:
		new_sample_size = self.sample_size + other.sample_size
		self.avg_loss = (other.avg_loss * other.sample_size + self.avg_loss * other.sample_size) / new_sample_size
		self.max_loss = max(self.max_loss, other.max_loss)
		self.min_loss = min(self.min_loss, other.min_loss)
		self.accuracy = (other.accuracy * other.sample_size + self.accuracy * self.sample_size) / new_sample_size if self.accuracy is not None and other.accuracy is not None else None
		self.sample_size = new_sample_size
		return self
	def __copy__(self) -> SampleCollection:
		return SampleCollection(self.avg_loss, self.max_loss, self.min_loss, self.accuracy, self.time, self.epoch, self.sample_size,)

class Metrics:
	def __init__(self, max_samples: int = 2**14) -> None:
		self._total_samples: int = 0
		self._max_samples: int = max_samples
		self._target_sample_size: int = 1
		self._samples: list[SampleCollection] = []
		self._total_time: float = 0
	def record(self, sample: SampleCollection) -> None:
		if self._samples[-1].sample_size < self._target_sample_size:
			self._samples[-1].merge(sample)
			self._last_sample_size += self._samples[-1].sample_size
		else:
			self._samples.append(sample)
			self._last_sample_size = self._samples[-1].sample_size
		if len(self._samples) > self._max_samples:
			self._samples = [(self._samples[i].merge(self._samples[i + 1]) if i + 1 < len(self._samples) else self._samples[i]) for i in range(0, len(self._samples), 2)]
			self._target_sample_size *= 2
		self._total_samples += 1
	def __getitem__(self, position: int | float) -> SampleCollection:
		return self._samples[self._get_index(position)]
	def merge_range(self, start: int | float, end: int | float) -> SampleCollection:
		start_index = self._get_index(start)
		end_index = self._get_index(end)
		if start_index > end_index:
			raise ValueError("Invalid range")
		output = copy(self._samples[start_index]) 
		for i in range(start_index + 1, end_index):
			output.merge(self._samples[i])
		return output
	def _get_index(self, position: int | float) -> int:
		if isinstance(position, int):
			return int(position / self._total_samples * len(self._samples))
		else:
			return int(self._total_samples * position)
