from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID

from abc import ABC as Abstract, abstractmethod

from copy import copy

def or_search(schema: Schema, evaluator: Evaluator, selector: Selector, max_id: ID | int, model_pool_size: int = 1, breed_iterations: int = 1) -> ModelPool:
	indices = BreedIndices()
	model_pool: ModelPool = [] 
	i = 0
	while i < breed_iterations: #will switch this to use a call back? allowing for an interactive cli?
		print(f"Breeding iteration {i} (this will be taken away when better logging is implemented)")
		for j in range(model_pool_size):
			if (ir := schema.compile_ir(evaluator.get_input_shapes(), indices, max_id)) is not None:
				training_metrics, validation_metrics = evaluator.evaluate(ir)
				model_pool.append((ir, training_metrics, validation_metrics))
			else:
				raise ValueError("Failed compilation")
		model_pool = selector.select(model_pool, model_pool_size)
		indices = BreedIndices([ir for ir, _, _ in model_pool], .2, .2, .2) 
		i += 1
	return model_pool

class Selector(Abstract):
	@abstractmethod
	def select(self, models: ModelPool, model_pool_size: int) -> ModelPool:
		pass

class AvgEpochLossSelector(Selector):
	def select(self, models: ModelPool, model_pool_size: int) -> ModelPool:
		raise NotImplementedError

class AvgLossWindowSelector(Selector):
	def __init__(self, window_size: int) -> None:
		self._window_size = window_size
	def select(self, models: ModelPool, model_pool_size: int) -> ModelPool:
		scores= []
		for model in models:
			_, training_metrics, validation_metrics = model
			focused_metrics = training_metrics if validation_metrics is None else validation_metrics
			start_index, end_index = 0, 0
			loss = 0
			min_loss = float("inf")
			samples = 0
			while end_index < len(focused_metrics.get_sample_list()):
				while samples < self._window_size and end_index < len(focused_metrics.get_sample_list()):
					loss += focused_metrics[end_index].total_loss
					samples += focused_metrics[end_index].sample_size
					end_index += 1
				if loss / samples < min_loss:
					min_loss = loss / samples
				while samples >= self._window_size:
					loss -= focused_metrics[start_index].total_loss
					samples -= focused_metrics[start_index].sample_size
					start_index += 1
			scores.append((min_loss, model))
		scores.sort(key=lambda pair: pair[0])
		return [model for _, model in scores[:model_pool_size]]

class Evaluator(Abstract):
	@abstractmethod
	def evaluate(self, ir: list[IRNode]) -> tuple[Metrics, Metrics | None]:
		pass
	@abstractmethod
	def get_input_shapes(self) -> list[LockedShape]:
		pass

class SampleCollection:
	__slots__ = ["sample_size", "total_loss", "max_loss", "min_loss", "correct", "time", "epoch"]
	def __init__(self, total_loss: float, max_loss: float, min_loss: float, total_correct: float | None, time: float | None, epoch: int | None, sample_size: int = 1) -> None:
		self.sample_size: int = sample_size 
		self.total_loss: float = total_loss
		self.max_loss: float = max_loss
		self.min_loss: float = min_loss 
		self.correct: float | None = total_correct 
		self.time: float | None = time 
		self.epoch: int | None = epoch
	def merge(self, other: SampleCollection) -> SampleCollection:
		return SampleCollection(
			self.total_loss + other.total_loss,
			max(self.max_loss, other.max_loss),
			min(self.min_loss, other.min_loss),
			self.correct + other.correct if self.correct is not None and other.correct is not None else None,
			self.time + other.time if self.time is not None and other.time is not None else None,
			self.epoch,
			self.sample_size + other.sample_size,
		)
	def __copy__(self) -> SampleCollection:
		return SampleCollection(self.total_loss, self.max_loss, self.min_loss, self.correct, self.time, self.epoch, self.sample_size,)
	def __str__(self) -> str:
		return (f"loss: {self.total_loss / self.sample_size}, max: {self.max_loss}, min: {self.min_loss}"
			+ f", accuracy: {self.correct / self.sample_size}" if self.correct is not None else ""
			+ f", sample size: {self.sample_size}"
			+ f", time: {self.time}"	if self.time is not None else ""
		  	+ f", epoch: {self.epoch}" if self.epoch is not None else "")
	def __repr__(self) -> str:
		return str(self)
class Metrics:
	def __init__(self, max_samples: int = 2**14) -> None:
		self._total_samples: int = 0
		self._max_samples: int = max_samples
		self._target_sample_size: int = 1
		self._samples: list[SampleCollection] = []
		self._total_time: float = 0
	def record(self, sample: SampleCollection) -> None:
		if len(self._samples) > 0 and self._samples[-1].sample_size < self._target_sample_size:
			self._samples[-1] = self._samples[-1].merge(sample)
			self._last_sample_size += self._samples[-1].sample_size
		else:
			self._samples.append(sample)
			self._last_sample_size = self._samples[-1].sample_size
		if len(self._samples) > self._max_samples:
			self._samples = [(self._samples[i].merge(self._samples[i + 1]) if i + 1 < len(self._samples) else self._samples[i]) for i in range(0, len(self._samples), 2)]
			self._target_sample_size *= 2
		self._total_samples += 1
		if self._samples[-1].sample_size < self._target_sample_size:
			self._samples[-1].merge(sample)
			self._last_sample_size += self._samples[-1].sample_size
	def get_epochs(self) -> list[SampleCollection]:
		return []
	def __getitem__(self, position: int | float) -> SampleCollection:
		return self._samples[self._get_index(position)]
	def merge_range(self, start: int | float, end: int | float) -> SampleCollection:
		start_index = self._get_index(start)
		end_index = self._get_index(end)
		if start_index > end_index:
			raise ValueError("Invalid range")
		output = self._samples[start_index] 
		for i in range(start_index + 1, end_index):
			output = output.merge(self._samples[i])
		return output
	def _get_index(self, position: int | float) -> int:
		index = 0
		if isinstance(position, int):
			index = int(position / self._total_samples * len(self._samples))
		else:
			index = int(self._total_samples * position)
		return min(index, len(self._samples) - 1)
	def format(self, resolution: int | None) -> str:
		if resolution is None:
			resolution = len(self._samples)
		return "\n".join([f"{self[i/resolution]}" for i in range(10)])
	def get_sample_list(self) -> list[SampleCollection]:
		return self._samples
	def __str__(self) -> str:
		return self.format(20)
	def __repr__(self) -> str:
		return self.format(20)

ModelPool = list[tuple[list[IRNode], Metrics, Metrics | None]]
