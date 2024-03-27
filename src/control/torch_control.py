#
#
# TEMPORARY, will be replaced with generic control, taking framework adapters
#
#

from __future__ import annotations

from ..schema import Schema, BreedIndices, IRNode
from ..shared import LockedShape, ID
from ..adapter import get_module, DefaultComponentFormatter, generate_torch_module

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch import Tensor
import torch

from pandas import DataFrame

from typing import Any, Callable
from enum import Enum

from random import random

class OptimizerType(Enum):
	ADAM = "adam"
	SGD = "sgd"
class CompileBackend(Enum):
	INDUCTOR = "inductor"
	CUDA_GRAPHS = "cudagraphs"
CUDA = "cuda"
CPU = "cpu"
AccuracyFunction = Callable[[Tensor, Tensor], float]

class Control:
	def __init__(self, 
			schema: Schema, 
			train_dataset: Dataset, 
			validation_dataset: Dataset, 
			max_id: ID = ID(1024),
			compile_models: bool = True, 
			compiler_backend: CompileBackend = CompileBackend.INDUCTOR, 
			require_cuda: bool = True,
			accuracy_function: AccuracyFunction = lambda x, y: 0,
			) -> None:
		self._schema: Schema = schema
		self._train_dataset: Dataset = train_dataset
		self._validation_dataset: Dataset = validation_dataset
		self._max_id: ID = max_id
		self._compile_models: bool = compile_models
		self._compiler_backend: CompileBackend = compiler_backend
		self._require_cuda: bool = require_cuda
		self._accuracy_function: AccuracyFunction = accuracy_function
	def search(self, 
			input_shapes: list[LockedShape], 
			save_dir: str, 
			criterion: Module, 
			optimizer_type: OptimizerType = OptimizerType.ADAM, 
			batch_size: int = 5, 
			workers: int = 0, 
			model_pool_size: int = 1,
			training_epochs: int = 1,
			breed_iterations: int = 1,
			validation_multiple: int = 5
		) -> None:
		#look into taking in a sampler for the data loader, may be useful for large datasets
		device_type = CUDA if torch.cuda.is_available() else CPU 
		if self._require_cuda and not device_type == CUDA:
			raise ValueError("CUDA set to required but not available")
		device = torch.device(device_type)
		train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		validation_loader = DataLoader(self._validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
		test_indices: list[BreedIndices] = [BreedIndices() for _ in range(model_pool_size)]
		model_pool: list[ModelTracker] = [] 
		failed_compilations = 0
		i = 0
		while i < breed_iterations: #will switch this to use a call back? allowing for a cli?
			for j, indices in enumerate(test_indices):
				if (ir := self._schema.compile_ir(input_shapes, indices, self._max_id)) is not None:
					model = ModelTracker(ir)
					runnable_model: Any = get_module(f"M{i}_{j}", ir, DefaultComponentFormatter())
					if self._compile_models:
						runnable_model = torch.compile(runnable_model, backend=str(self._compiler_backend)) 
					runnable_model.to(device)
					optimizer = get_optimizer(optimizer_type, runnable_model)
					for j in range(training_epochs):
						print("Epoch", j)
						model.record_training_epoch(train_epoch(runnable_model, criterion, self._accuracy_function, optimizer, train_loader, device, device_type))
						if (j + 1) % validation_multiple == 0:
							model.record_validation_epoch(validate_epoch(runnable_model, criterion, self._accuracy_function, validation_loader, device, device_type))
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
		with open(f"{save_dir}/model_{i}.py", "w") as file:
			file.write(generate_torch_module(f"M{i}", model.get_ir()))
		DataFrame({"accuracy": [metrics.get_accuracy() for metrics in model._train], "loss": [metrics.get_loss() for metrics in model._train]}).to_csv(f"{save_dir}/model_{i}_train.csv")
		DataFrame({"accuracy": [metrics.get_accuracy() for metrics in model._validation], "loss": [metrics.get_loss() for metrics in model._validation]}).to_csv(f"{save_dir}/model_{i}_validation.csv")
	return model_pool
def train_epoch(model: Module, criterion: Module, accuracy_function: AccuracyFunction, optimizer: torch.optim.Optimizer, train_loader: DataLoader, device: torch.device, device_type: str) -> EpochMetrics:
	metrics: EpochMetrics = EpochMetrics()
	model.train()
	scaler = torch.cuda.amp.GradScaler()
	for i, (data, truth) in enumerate(train_loader):
		data, truth = data.to(device), truth.to(device)
		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device_type, dtype=torch.float16):
			output = model(data)
			loss = criterion(output, truth)
			accuracy = accuracy_function(output, truth)
			metrics.record(loss.item(), accuracy)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
	print(metrics)
	return metrics 
def validate_epoch(model: Module, criterion: Module, accuracy_function: AccuracyFunction, validation_loader: DataLoader, device: torch.device, device_type: str) -> EpochMetrics:
	metrics: EpochMetrics = EpochMetrics()
	model.eval()
	with torch.no_grad():
		for data, truth in validation_loader:
			data, truth = data.to(device), truth.to(device)
			with torch.autocast(device_type=device_type, dtype=torch.float16):
				output = model(data)
				loss = criterion(output, truth)
				accuracy = accuracy_function(output, truth)
				metrics.record(loss.item(), accuracy)
	print(metrics)
	return metrics 


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

def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = learning_rate
def get_optimizer(optimizer_type: OptimizerType, model: Any) -> torch.optim.Optimizer:
	if optimizer_type == OptimizerType.ADAM:
		return torch.optim.Adam(model.parameters(), lr=0.0002)
	elif optimizer_type == OptimizerType.SGD:
		return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	else:
		raise ValueError("Invalid optimizer type")
