from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID

import csv

from abc import ABC as Abstract, abstractmethod

from random import random

class Control:
	def __init__(self, 
			schema: Schema, 
			runner: Runner,
		) -> None:
		self._schema: Schema = schema
		self._runner: Runner = runner
	def search(self, 
			save_dir: str, 
			model_pool_size: int = 1,
			breed_iterations: int = 1,
		) -> None:
		test_indices: list[BreedIndices] = [BreedIndices() for _ in range(model_pool_size)]
		model_pool: list[ModelTracker] = [] 
		failed_compilations = 0
		i = 0
		while i < breed_iterations: #will switch this to use a call back? allowing for an interactive cli?
			for j, indices in enumerate(test_indices):
				if (ir := self._schema.compile_ir(self._input_shapes, indices, self._max_id)) is not None:
					stats = self._runner.evaluate_model(ir)
					model = ModelTracker(ir)
					runnable_model = self._runner_builder.build(ir)
					for j in range(training_epochs):
						print("Epoch", j)
						model.record_training_epoch(runnable_model.train_epoch())
						if (j + 1) % validation_multiple == 0:
							model.record_validation_epoch(runnable_model.validate_epoch())
					model_pool.append(model)
				else:
					failed_compilations += 1
					if failed_compilations > 10:
						raise ValueError("Too many failed compilations")
				model_pool = cull_and_save_models(model_pool, model_pool_size, save_dir)
			test_indices = [BreedIndices([tracker.get_ir() for tracker in model_pool if random() < .2], .2, .2, .2) for _ in range(model_pool_size)]
			i += 1



def cull_and_save_models(model_pool: list[ModelTracker], max_pool_size: int, save_dir: str) -> list[ModelTracker]:
	model_pool.sort(key=lambda model: model.get_min_validation_loss())
	model_pool = model_pool[:max_pool_size]
	for i, model in enumerate(model_pool):
		#with open(f"{save_dir}/model_{i}.py", "w") as file:
		#	file.write(generate_torch_module(f"M{i}", model.get_ir()))
		#DataFrame({"accuracy": [metrics.get_accuracy() for metrics in model._train], "loss": [metrics.get_loss() for metrics in model._train]}).to_csv(f"{save_dir}/model_{i}_train.csv")
		#DataFrame({"accuracy": [metrics.get_accuracy() for metrics in model._validation], "loss": [metrics.get_loss() for metrics in model._validation]}).to_csv(f"{save_dir}/model_{i}_validation.csv")
		pass
	return model_pool

class Runner(Abstract):
	@abstractmethod
	def evaluate_model(self, ir: list[IRNode]) -> ModelTracker:
		pass
	@abstractmethod
	def get_input_shapes(self) -> list[LockedShape]:
		pass

class ModelTracker:
	__slots__ = ["_train", "_validation", "_ir"]
	def __init__(self, ir: list[IRNode]) -> None:
		self._ir: list[IRNode] = ir
		self._train: list[EpochMetrics] = []
		self._validation: list[EpochMetrics] = []
	def record_training_data(self, epoch: int, loss: float, correct: float, samples: int = 1) -> None: #may splits eqoch
		if len(self._train) <= epoch:
			self._train.append(EpochMetrics())
		self._train[epoch].record(loss, correct, samples)
	def record_validation_data(self, epoch: int, loss: float, correct: float, samples: int = 1) -> None:
		if len(self._validation) <= epoch:
			self._validation.append(EpochMetrics())
		self._validation[epoch].record(loss, correct, samples)
	def record_training_epoch(self, metrics: EpochMetrics) -> None:
		self._train.append(metrics)
	def record_validation_epoch(self, metrics: EpochMetrics) -> None:
		self._validation.append(metrics)
	def get_min_validation_loss(self) -> float:
		return min([x.get_loss() for x in self._validation])
	def get_min_training_loss(self) -> float:
		return min([x.get_loss() for x in self._train])
	def get_ir(self) -> list[IRNode]:
		return self._ir
class EpochMetrics:
	__slots__ = ["_loss_total", "loss_max", "loss_min", "_correct_total", "_samples"]
	def __init__(self) -> None:
		self._loss_total: float = 0
		self.loss_max: float = float("-inf")
		self.loss_min: float = float("inf")
		self._correct_total: float = 0
		self._samples: int = 0
	def record(self, loss: float, correct: float, samples: int = 1) -> None:
		self._samples += samples 
		self._loss_total += loss
		self.loss_max = max(self.loss_max, loss)
		self.loss_min = min(self.loss_min, loss)
		self._correct_total += correct 
	def get_loss(self) -> float:
		return self._loss_total / self._samples
	def get_accuracy(self) -> float:	
		return self._correct_total / self._samples
	def __str__(self) -> str:
		return f"Loss: {self.get_loss()}, Accuracy: {self.get_accuracy()}"
	def __repr__(self) -> str:
		return str(self)
