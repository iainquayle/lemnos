from __future__ import annotations

from ...schema import IRNode
from ...control import Evaluator 
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

class BasicLossBased(Evaluator):
	def __init__(self, 
			formatter: TorchComponentFormatter,
			train_loader: DataLoader,
			validation_loader: DataLoader | None,
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
		self._train_loader = train_loader 
		self._validation_loader = validation_loader
		self._criterion = criterion
		self._accuracy_function = accuracy_function
		self._lr = lr
		self._optimizer = optimizer
		self._device = device
		self._torch_compiler = torch_compiler
	def evaluate_model(self, ir: list[IRNode]) -> float:
		#impl record, and use it for this
		#perhaps make this basic something... then allow the user to define the selector based on the metrics passed back 
		model: Any = create_module("Model", ir, self._formatter)
		if self._torch_compiler is not None:
			model = torch.compile(model, backend=str(self._torch_compiler))
		model.to(self._device)
		optimizer = self._optimizer.get(model)
		model.train()
		scaler = torch.cuda.amp.GradScaler()
		for (input, truth) in self._train_loader:
			input, truth = input.to(self._device), truth.to(self._device)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type=self._device_type, dtype=torch.float16):
				output = model(input)
				loss = self._criterion(output, truth)
				if self._accuracy_function is not None:
					accuracy = self._accuracy_function(output, truth)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			gc.collect()
		return 0 
		
class Sample(Abstract):
	@abstractmethod
	def combine(self, other: Sample) -> Sample:
		pass

class BasicSample(Sample):
	def __init__(self) -> None:
		self.avg_loss: float = 0
		self.max_loss: float = 0
		self.min_loss: float = 0
		pass
	def combine(self, other: Sample) -> Sample:
		return BasicSample()

#could techincally make a memory backed record, however it still has the potential to absolutely dummy memory if not careful
class Record:
	def __init__(self, max_samples: int = 2**14) -> None:
		self._total_samples: int = 0
		self._max_samples: int = max_samples
		self._samples: list[Sample] = []
		self._total_time: float = 0
	def record(self, sample: Sample) -> None:
		self._total_samples += 1
		self._samples.append(sample)
		if len(self._samples) > self._max_samples:
			self._samples = [self._samples[i].combine(self._samples[i + 1]) for i in range(0, len(self._samples), 2)]
	def __getitem__(self, position: int | float) -> Sample:
		index = 0
		if isinstance(position, int):
			index = int(position / self._total_samples * len(self._samples))
		else:
			index = int(self._total_samples * position)
		return self._samples[index]
		

class CappedQueue: 
	def __init__(self, size: int) -> None:
		self._size: int = size
		self._pointer: int = 0
		self._full: bool = False
		self._queue: list[Any] = [0 for _ in range(size)]
	def push(self, item: Any) -> None:
		self._queue[self._pointer] = item
		self._pointer += 1
		if self._pointer == self._size:
			self._full = True
			self._pointer = 0
	def __getitem__(self, index: int) -> Any:
		return self._queue[index]
	def __len__(self) -> int:
		return self._size if self._full else self._pointer

def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = learning_rate
