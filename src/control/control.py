from __future__ import annotations

from ..schema import Schema, BreedIndices
from ..model import Model
from ..shared import LockedShape

from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch

from typing import List, Tuple

from enum import Enum

class OptimizerType(Enum):
	ADAM = "adam"
	SGD = "sgd"

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
	def search(self, input_shapes: List[LockedShape], save_dir: str, criterion: Module, optimizer_type: OptimizerType = OptimizerType.ADAM, batch_size: int = 5, workers: int = 0, model_pool_size: int = 1) -> None:
		#look into taking in a sampler for the data loader, may be useful for large datasets
		device_type = CUDA if torch.cuda.is_available() else CPU 
		device = torch.device(device_type)
		train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		validation_loader = DataLoader(self._validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
		model_pool: List[Model] = []
		for _ in range(model_pool_size):
			if (model := self._schema.build(input_shapes, BreedIndices())) is not None:
				model_pool.append(model)
			else:
				raise ValueError("Model could not be built from schema")
		model_training_stats: List[List[Tuple[float, float]]] = [[] for _ in model_pool]
		breed_iterations = 1
		for i in range(breed_iterations):
			for model in model_pool:
				runnable_model = model.get_torch_module_handle(f"M{i}")()
				if COMPILE_MODELS:
					runnable_model = torch.compile(runnable_model, backend=COMPILER_BACKEND) 
				runnable_model.to(device)
				epochs = 1
				#add another loop here? or validate on every epoch? 
				training_loss_avg: float = 0
				validation_loss: float = 0
				for epoch in range(epochs):
					runnable_model.train()
					optimizer = torch.optim.Adam(runnable_model.parameters())
					scaler = torch.cuda.amp.GradScaler()
					for data, truth in train_loader:
						data, truth = data.to(device), truth.to(device)
						optimizer.zero_grad(set_to_none=True)
						with torch.autocast(device_type=device_type, dtype=torch.float16):
							output = runnable_model(data)
							loss = criterion(output, truth)
							
						scaler.scale(loss).backward()
						scaler.step(optimizer)
						scaler.update()
				for data, truth in validation_loader:
					runnable_model.eval()
					for data, truth in validation_loader:
						data, truth = data.to(device), truth.to(device)
						with torch.no_grad():
							output = runnable_model(data)
							loss = criterion(output, truth)
							model_training_stats[i].append((loss.item(), 0))
					pass
