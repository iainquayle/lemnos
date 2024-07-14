from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID

from abc import ABC as Abstract, abstractmethod

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
	def __init__(self, sample_size: int) -> None:
		self._sample_size = sample_size
	def select(self, models: ModelPool, model_pool_size: int) -> ModelPool:
		scores= []
		for model in models:
			_, training_metrics, validation_metrics = model
			focused_metrics = training_metrics if validation_metrics is None else validation_metrics
			start_index, end_index = 0, 0
			loss = 0
			min_loss = float("inf")
			samples = 0
			while end_index < len(focused_metrics):
				while samples < self._sample_size and end_index < len(focused_metrics):
					loss += focused_metrics[end_index].total_loss
					samples += focused_metrics[end_index].sample_size
					end_index += 1
				if loss / samples < min_loss:
					min_loss = loss / samples
				while samples >= self._sample_size:
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

class ResultsSample:
	__slots__ = ["sample_size", "total_loss", "max_loss", "min_loss", "correct", "time", "epoch"]
	def __init__(self, loss: float, max_loss: float, min_loss: float, total_correct: float | None, time: float | None, epoch: int | None, sample_size: int = 1) -> None:
		self.sample_size: int = sample_size 
		self.total_loss: float = loss * sample_size 
		self.max_loss: float = max_loss
		self.min_loss: float = min_loss 
		self.correct: float | None = total_correct 
		self.time: float | None = time 
		self.epoch: int | None = epoch
	def merge(self, other: ResultsSample) -> ResultsSample:
		new_collection = ResultsSample(
			(self.total_loss + other.total_loss) / (self.sample_size + other.sample_size),
			max(self.max_loss, other.max_loss),
			min(self.min_loss, other.min_loss),
			self.correct + other.correct if self.correct is not None and other.correct is not None else None,
			self.time + other.time if self.time is not None and other.time is not None else None,
			max(self.epoch, other.epoch) if self.epoch is not None and other.epoch is not None else None,
			self.sample_size + other.sample_size,
		)
		return new_collection
	def get_loss(self) -> float:
		return self.total_loss / self.sample_size
	def get_accuracy(self) -> float | None:
		return self.correct / self.sample_size if self.correct is not None else None
	def __copy__(self) -> ResultsSample:
		return ResultsSample(self.total_loss, self.max_loss, self.min_loss, self.correct, self.time, self.epoch, self.sample_size,)
	def __str__(self) -> str:
		return (f"loss: {self.get_loss()}, max: {self.max_loss}, min: {self.min_loss}"
			+ (f", accuracy: {self.correct / self.sample_size}" if self.correct is not None else "")
			+ f", sample size: {self.sample_size}"
			+ (f", time: {self.time}" if self.time is not None else "")
		  	+ (f", epoch: {self.epoch}" if self.epoch is not None else ""))
	def __repr__(self) -> str:
		return str(self)
class Metrics:
	def __init__(self, max_resolution: int = 2**10) -> None:
		self._total_samples: int = 0
		self._max_resolution: int = max_resolution
		self._target_sample_size: int = 1
		self._samples: list[ResultsSample] = []
	def record(self, sample: ResultsSample) -> None:
		if len(self._samples) == 0:
			self._samples.append(sample)
			self._last_sample_size = sample.sample_size 
			self._total_samples += sample.sample_size
			self._target_sample_size = sample.sample_size
		else:
			if self._samples[-1].sample_size < self._target_sample_size:
				self._samples[-1] = self._samples[-1].merge(sample)
				self._last_sample_size += self._samples[-1].sample_size
			else:
				self._samples.append(sample)
				self._last_sample_size = self._samples[-1].sample_size
			if len(self._samples) > self._max_resolution:
				self._samples = [(self._samples[i].merge(self._samples[i + 1]) if i + 1 < len(self._samples) else self._samples[i]) for i in range(0, len(self._samples), 2)]
				self._target_sample_size *= 2
			self._total_samples += sample.sample_size 
	def get_epochs(self) -> list[ResultsSample]:
		raise NotImplementedError
	def get_total_samples(self) -> int:
		return self._total_samples
	def __getitem__(self, index: int) -> ResultsSample:
		return self._samples[index]
	def __len__(self) -> int:
		return len(self._samples)
	def get_range_by_samples(self, start_sample: int, end_sample: int) -> ResultsSample:
		start_index = self._get_sample_index(start_sample)
		end_index = self._get_sample_index(end_sample)
		if start_index == end_index:
			return self._samples[start_index]
		return self.get_range(start_index, end_index)
	def get_range(self, start_index: int, end_index: int) -> ResultsSample:
		start_index = self._get_index(start_index)
		end_index = self._get_index(end_index)
		output = self._samples[start_index] 
		for i in range(start_index + 1, end_index):
			output = output.merge(self._samples[i])
		return output
	def get_fractional(self, position: float) -> ResultsSample:
		return self._samples[int(len(self._samples) * position)]
	def format(self, resolution: int | None) -> str:
		if resolution is None:
			resolution = len(self._samples)
		step = len(self._samples) / resolution
		return "\n".join((f"{self.get_range(int(i * step), int((i + 1) * step) - 1)}" for i in range(resolution)))
	def __str__(self) -> str:
		return self.format(20)
	def __repr__(self) -> str:
		return self.format(20)
	def _get_index(self, index: int) -> int:
		return index if index >= 0 else len(self._samples) + index
	def _get_sample_index(self, sample_index: int) -> int:
		sample_index = sample_index if sample_index >= 0 else self._total_samples + sample_index
		return int(sample_index / self._total_samples * len(self._samples))
		 

ModelPool = list[tuple[list[IRNode], Metrics, Metrics | None]]
