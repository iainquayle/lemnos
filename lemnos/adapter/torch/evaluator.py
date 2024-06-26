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
				if self._training_example_count % 2**10 == 0:
					print(f"example count: {self._training_example_count} * batch size")
					print(training_metrics)
				gc.collect()
			if self._validation_loader is not None:
				model.eval()
				with torch.no_grad():
					for (input, truth) in self._validation_loader:
						input, truth = input.to(device), truth.to(device)
						with torch.autocast(device_type=self._device_type, dtype=torch.float16):
							output = model(input)
							loss = self._criterion(output, truth)
							accuracy = self._accuracy_function(output, truth) if self._accuracy_function is not None else None
						validation_metrics.record(SampleCollection(loss.item(), loss.item(), loss.item(), accuracy, None, epoch))
						gc.collect()
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
