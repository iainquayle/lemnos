from __future__ import annotations

from ..schema import Schema, BreedIndices
from ..model import Model
from ..shared import LockedShape

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch

from typing import List, Tuple, Any
from enum import Enum
from dataclasses import dataclass

class OptimizerType(Enum):
	ADAM = "adam"
	SGD = "sgd"

REQUIRE_CUDA = True

CUDA = "cuda"
CPU = "cpu"
INDUCTOR = "inductor"
CUDA_GRAPHS = "cudagraphs"

COMPILE_MODELS = True 
COMPILER_BACKEND = INDUCTOR

class Control:
	def __init__(self, schema: Schema, train_dataset: Dataset, validation_dataset: Dataset) -> None:
		self._schema: Schema = schema
		self._train_dataset: Dataset = train_dataset
		self._validation_dataset: Dataset = validation_dataset
		#should also hold onto data transformation modules? though this could technically be done in the dataset class?
	def search(self, 
			input_shapes: List[LockedShape], 
			save_dir: str, 
			criterion: Module, 
			optimizer_type: OptimizerType = OptimizerType.ADAM, 
			batch_size: int = 5, 
			workers: int = 0, 
			model_pool_size: int = 1
		) -> None:
		#look into taking in a sampler for the data loader, may be useful for large datasets
		device_type = CUDA if torch.cuda.is_available() else CPU 
		if REQUIRE_CUDA and not device_type == CUDA:
			raise ValueError("CUDA set to required but not available")
		device = torch.device(device_type)
		train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		validation_loader = DataLoader(self._validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
		test_models: List[Model] = []
		for _ in range(model_pool_size):
			if (model := self._schema.build(input_shapes, BreedIndices())) is not None:
				test_models.append(model)
			else:
				raise ValueError("Model could not be built from schema")
		#could make the performance stats hold the model instead?
		model_pool: List[Tuple[Model, List[PerformanceStats], List[PerformanceStats]]] = []
		breed_iterations = 1
		for i in range(breed_iterations):
			for model in test_models:
				model_pool.append((model, [], []))
				runnable_model = model.get_torch_module_handle(f"M{i}")()
				if COMPILE_MODELS:
					runnable_model = torch.compile(runnable_model, backend=COMPILER_BACKEND) 
				runnable_model.to(device)
				optimizer = get_optimizer(optimizer_type, runnable_model)
				epochs = 1
				for _ in range(1):
					runnable_model.train()
					training_stats: PerformanceStats = PerformanceStats()
					for _ in range(epochs):
						training_stats = PerformanceStats()
						scaler = torch.cuda.amp.GradScaler()
						for data, truth in train_loader:
							data, truth = data.to(device), truth.to(device)
							optimizer.zero_grad(set_to_none=True)
							with torch.autocast(device_type=device_type, dtype=torch.float16):
								output = runnable_model(data)
								loss = criterion(output, truth)
								training_stats.record(loss.item(), 0)
							scaler.scale(loss).backward()
							scaler.step(optimizer)
							scaler.update()
					runnable_model.eval()
					model_pool[-1][1].append(training_stats)
					validation_stats: PerformanceStats = PerformanceStats()
					for data, truth in validation_loader:
						data, truth = data.to(device), truth.to(device)
						with torch.no_grad():
							output = runnable_model(data)
							loss = criterion(output, truth)
							validation_stats.record(loss.item(), 0)
					model_pool[-1][2].append(validation_stats)

def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = learning_rate
def get_optimizer(optimizer_type: OptimizerType, model: Any) -> torch.optim.Optimizer:
	if optimizer_type == OptimizerType.ADAM:
		return torch.optim.Adam(model.parameters())
	elif optimizer_type == OptimizerType.SGD:
		return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	else:
		raise ValueError("Invalid optimizer type")

class PerformanceStats:
	__slots__ = ["_loss_total", "loss_max", "loss_min", "_correct_total", "_samples"]
	def __init__(self) -> None:
		self._loss_total: float
		self.loss_max: float
		self.loss_min: float
		self._correct_total: float
		self._samples: int
	def record(self, loss: float, correct: float, samples: int = 1) -> None:
		self._samples += samples 
		self._loss_total += loss
		self.loss_max = max(self.loss_max, loss)
		self.loss_min = min(self.loss_min, loss)
		self._correct_total += correct 
	def get_average_loss(self) -> float:
		return self._loss_total / self._samples
	#this will need to have a user defined function to calculate whether something is correct or not 
	#thus will leave it alone rn
	#def get_accuracy(self) -> float:	
	#	return self._correct_total / self._samples
