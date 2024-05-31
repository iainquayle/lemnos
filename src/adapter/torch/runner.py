from __future__ import annotations

from ...schema import IRNode
from ...control import Runner 
from .formatter import DefaultComponentFormatter, TorchComponentFormatter, create_module 

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, IterableDataset, Dataset

from typing import Callable, Any

from enum import Enum
from abc import ABC as Abstract, abstractmethod

import gc

class CompileBackend(Enum):
	INDUCTOR = "inductor"
	CUDA_GRAPHS = "cudagraphs"

CUDA = "cuda"
CPU = "cpu"

class Optimizer(Abstract):
	@abstractmethod
	def get(self, model: Any) -> torch.optim.Optimizer:
		pass
class Adam(Optimizer):
	def __init__(self, lr: float) -> None:
		self._lr = lr
	def get(self, model: Any) -> torch.optim.Optimizer:
		return torch.optim.Adam(model.parameters(), lr=self._lr)
class SGD(Optimizer):
	def __init__(self, lr: float, momentum: float) -> None:
		self._lr = lr
		self._momentum = momentum
	def get(self, model: Any) -> torch.optim.Optimizer:
		return torch.optim.SGD(model.parameters(), lr=self._lr, momentum=self._momentum)

AccuracyFunction = Callable[[Tensor, Tensor], float]
class TorchRunnerBuilder(RunnerBuilder):
	def __init__(self,
			train_dataset: Dataset,
			validation_dataset: Dataset,
			criterion: Module,
			accuracy_function: AccuracyFunction,
			lr: float = 0.0002,
			formatter: TorchComponentFormatter = DefaultComponentFormatter(),
			optimizer: Optimizer = Adam(lr=0.0002),
			batch_size: int = 32,
			workers: int = 0,
			require_cuda: bool = False,
			compiler_backend: CompileBackend | None = CompileBackend.INDUCTOR,
		) -> None:
		self._device_type = CUDA if torch.cuda.is_available() else CPU 
		if require_cuda and not self._device_type == CUDA:
			raise ValueError("CUDA set to required but not available")
		self._device = torch.device(self._device_type)
		self._train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		self._validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
		self._criterion = criterion
		self._accuracy_function = accuracy_function
		self._lr = lr
		self._formatter = formatter
		self._compiler_backend = compiler_backend
		self._optimizer = optimizer 
	def build(self, ir: list[IRNode]) -> Runner:
		runnable_model: Any = create_module(f"Model", ir, DefaultComponentFormatter())
		if self._compiler_backend is not None:
			runnable_model = torch.compile(runnable_model, backend=str(self._compiler_backend)) 
		runnable_model.to(self._device)
		optimizer = self._optimizer.get(runnable_model)
		return TorchRunner(runnable_model, self._train_loader, self._validation_loader, optimizer, self._criterion, self._accuracy_function, self._device, self._device_type)

#change these to be just one? seems like a lot of work otherwise?
class TrainOnly(Runner):
	def __init__(self, 
			formatter: TorchComponentFormatter,
			data_loader: DataLoader,
			criterion: Module,
			accuracy_function: AccuracyFunction | None,
			lr: float,
			optimizer: Optimizer,
			device: torch.device,
			require_cuda: bool,
			torch_compiler: CompileBackend | None = None,
		) -> None:
		self._device_type = CUDA if torch.cuda.is_available() else CPU 
		if require_cuda and not self._device_type == CUDA:
			raise ValueError("CUDA not available")
		self._formatter = formatter
		self._data_loader = data_loader 
		self._criterion = criterion
		self._accuracy_function = accuracy_function
		self._lr = lr
		self._optimizer = optimizer
		self._device = device
		self._torch_compiler = torch_compiler
	def evaluate_model(self, ir: list[IRNode]) -> float:
		model: Any = create_module("Model", ir, self._formatter)
		if self._torch_compiler is not None:
			model = torch.compile(model, backend=str(self._torch_compiler))
		model.to(self._device)
		optimizer = self._optimizer.get(model)
		model.train()
		scaler = torch.cuda.amp.GradScaler()
		for (data, truth) in self._data_loader:
			data, truth = data.to(self._device), truth.to(self._device)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type=self._device_type, dtype=torch.float16):
				output = model(data)
				loss = self._criterion(output, truth)
				if self._accuracy_function is not None:
					accuracy = self._accuracy_function(output, truth)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		return 0
		
class TrainAndValidate(Runner):
	pass

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
	gc.collect()
	return metrics 



def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = learning_rate
