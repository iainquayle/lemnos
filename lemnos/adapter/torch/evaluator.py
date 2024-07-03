from __future__ import annotations

from ...shared import LockedShape
from ...schema import IRNode
from ...control import Evaluator, Metrics, SampleCollection
from .formatter import DefaultComponentFormatter, TorchComponentFormatter, create_module 

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader 

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
	def __init__(self, lr: float, decay: float = 0.0) -> None:
		self._lr = lr
		self._decay = decay
	def get(self, model: Any) -> torch.optim.Optimizer:
		return torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._decay)
class SGD(Optimizer):
	def __init__(self, lr: float, momentum: float, decay: float = 0.0) -> None:
		self._lr = lr
		self._momentum = momentum
		self._decay = decay
	def get(self, model: Any) -> torch.optim.Optimizer:
		return torch.optim.SGD(model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=self._decay)

class Scheduler(Abstract):
	@abstractmethod
	def get(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
		pass
class StepLR(Scheduler):
	def __init__(self, step_size: int, gamma: float) -> None:
		self._step_size = step_size
		self._gamma = gamma
	def get(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self._step_size, gamma=self._gamma)
class OneCycleLR(Scheduler):
	def __init__(self, max_lr: float, total_steps: int) -> None:
		self._max_lr = max_lr
		self._total_steps = total_steps
	def get(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
		return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self._max_lr, total_steps=self._total_steps)

#accuracy should return the number of correct predictions, not the mean
AccuracyFunction = Callable[[Tensor, Tensor], float]

class TorchEvaluator(Evaluator):
	def __init__(self, 
			train_loader: DataLoader,
			validation_loader: DataLoader | None,
			epochs: int,
			criterion: Module,
			accuracy_function: AccuracyFunction | None,
			optimizer: Optimizer,
			scheduler: Scheduler | None,
			require_cuda: bool,
			input_shapes: list[LockedShape] | None = None,
			metrics_resolution: int = 2048,
			formatter: TorchComponentFormatter = DefaultComponentFormatter(),
			torch_compiler: CompileBackend | None = None,
		) -> None:
		self._device_type = CUDA if torch.cuda.is_available() else CPU 
		if require_cuda and not self._device_type == CUDA:
			raise ValueError("CUDA not available")
		self._train_loader = train_loader 
		self._validation_loader = validation_loader
		self._epochs = epochs
		self._criterion = criterion
		self._accuracy_function = accuracy_function
		self._optimizer = optimizer
		self._scheduler = scheduler
		self._input_shapes = input_shapes
		self._metrics_resolution = metrics_resolution
		self._formatter = formatter
		self._torch_compiler = torch_compiler
		self._training_example_count = 0
	def evaluate(self, ir: list[IRNode]) -> tuple[Metrics, Metrics | None]:
		device = torch.cuda.current_device() if self._device_type == CUDA else torch.device(CPU)
		training_metrics = Metrics(self._metrics_resolution)
		validation_metrics = Metrics(self._metrics_resolution)
		model: Any = create_module("Model", ir, self._formatter)
		if self._torch_compiler is not None:
			model = torch.compile(model, backend=str(self._torch_compiler))
		model.to(device)
		optimizer = self._optimizer.get(model)
		scheduler = self._scheduler.get(optimizer) if self._scheduler is not None else None
		model.train()
		scaler = torch.cuda.amp.GradScaler()
		for epoch in range(self._epochs):
			for (input, truth) in self._train_loader:
				input, truth = input.to(device), truth.to(device)
				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type=self._device_type, dtype=torch.float16):
					output = model(input)
					loss = self._criterion(output, truth)
					accuracy = self._accuracy_function(output, truth) if self._accuracy_function is not None else None
					training_metrics.record(SampleCollection(loss.item(), loss.item(), loss.item(), accuracy, None, epoch, len(input)))
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				self._training_example_count += 1
				#if training_metrics.get_total_samples() % 2**12 == 0:
				#	print(f"sample count: {training_metrics.get_total_samples()}")
				#	print(training_metrics)
				gc.collect()
			if scheduler is not None:
				scheduler.step()
			if self._validation_loader is not None:
				model.eval()
				with torch.no_grad():
					for (input, truth) in self._validation_loader:
						input, truth = input.to(device), truth.to(device)
						with torch.autocast(device_type=self._device_type, dtype=torch.float16):
							output = model(input)
							loss = self._criterion(output, truth)
							accuracy = self._accuracy_function(output, truth) if self._accuracy_function is not None else None
						validation_metrics.record(SampleCollection(loss.item(), loss.item(), loss.item(), accuracy, None, epoch, len(input)))
						gc.collect()
				print("Validation Metrics")
				print(validation_metrics)
		return training_metrics, validation_metrics if self._validation_loader is not None else None
	def get_input_shapes(self) -> list[LockedShape]:
		if self._input_shapes is not None:
			return self._input_shapes
		else:
			first = list(next(iter(self._train_loader))[0].shape[1:])
			return [LockedShape(*first)]
		
def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = learning_rate
