from __future__ import annotations

from ...schema import IRNode
from ...control import Runner, RunnerBuilder, EpochMetrics

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from typing import Callable

from enum import Enum

class CompileBackend(Enum):
	INDUCTOR = "inductor"
	CUDA_GRAPHS = "cudagraphs"
CUDA = "cuda"
CPU = "cpu"

AccuracyFunction = Callable[[Tensor, Tensor], float]
class TorchRunnerBuilder(RunnerBuilder):
	def __init__(self,
			train_dataset: Dataset,
			validation_dataset: Dataset,
			batch_size: int = 32,
			workers: int = 0,
			require_cuda: bool = False
		
		) -> None:
		self._device_type = CUDA if torch.cuda.is_available() else CPU 
		if require_cuda and not self._device_type == CUDA:
			raise ValueError("CUDA set to required but not available")
		self._device = torch.device(self._device_type)
		self._train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		self._validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	def build(self, ir: list[IRNode]) -> Runner:
		pass

class TorchRunner(Runner):
	def __init__(self) -> None:
		pass
	def train_epoch(self) -> EpochMetrics:
		pass
	def validate_epoch(self) -> EpochMetrics:
		pass


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
