from __future__ import annotations

from ..schema import Schema, BreedIndices
from ..model import Model
from ..shared import LockedShape

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
import torch

from typing import List

from enum import Enum

class OptimizerType(Enum):
	ADAM = "adam"
	SGD = "sgd"

class Control:
	def __init__(self, schema: Schema, train_dataset: Dataset, validation_dataset: Dataset) -> None:
		self._schema: Schema = schema
		self._train_dataset: Dataset = train_dataset
		self._validation_dataset: Dataset = validation_dataset
	#consider for the optimizer, taking in a function that creates the optimizer
	def optimize(self, input_shapes: List[LockedShape], save_dir: str, criterion: Module, optimizer_type: OptimizerType = OptimizerType.ADAM, batch_size: int = 5, workers: int = 0, gene_pool_size: int = 1) -> None:
		#look into taking in a sampler for the data loader
		#could take in a list of optimizers and criterions to all test
		#also consider looking into a memory profiler for automated batch sizing?
		device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
		device = torch.device(device_type)
		train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
		validation_loader = DataLoader(self._validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
		gene_pool: List[Model] = []
		for _ in range(gene_pool_size):
			if (model := self._schema.build(input_shapes, BreedIndices())) is not None:
				gene_pool.append(model)
			else:
				raise ValueError("Model could not be built from schema")

		breed_iterations = 1
		for i in range(breed_iterations):
			for model in gene_pool:
				runnable_model: Module = model.get_torch_module_handle(f"M{i}")()
				compiled_model = torch.compile(runnable_model)
				epochs = 1
				for epoch in range(epochs):
					runnable_model.train()
					optimizer = torch.optim.Adam(runnable_model.parameters())
					for data, truth in train_loader:
						with torch.autocast(device_type=device_type, dtype=torch.float16):
							output = compiled_model(data)
							loss = criterion(output, truth)
						pass
				for data, truth in validation_loader:
					pass
		pass
