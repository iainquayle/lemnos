from __future__ import annotations

from src.schema import Schema, BreedIndices
from src.model import Model
from src.shared import LockedShape

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from typing import List

class Control:
	def __init__(self, schema: Schema, train_dataset: Dataset, validation_dataset: Dataset) -> None:
		self._schema: Schema = schema
		self._train_dataset: Dataset = train_dataset
		self._validation_dataset: Dataset = validation_dataset
	def optimize(self, input_shapes: List[LockedShape], save_dir: str, gene_pool_size: int, optimizer: type[Optimizer], criterion: Module, batch_size: int, workers: int =0) -> None:
		#look into taking in a sampler for the data loader
		#could take in a list of optimizers and criterions to all test
		#also consider looking into a memory profiler for automated batch sizing?
		#train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)
		#validation_loader = DataLoader(self._validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

		gene_pool: List[Model] = []
		for _ in range(gene_pool_size):
			if (model := self._schema.build(input_shapes, BreedIndices())) is not None:
				gene_pool.append(model)
			else:
				raise ValueError("Model could not be built from schema")

		breed_iterations = 1
		for i in range(breed_iterations):
			for model in gene_pool:
				epochs = 1
				runnable_model: Module = model.get_torch_module_handle(f"M{i}_{epochs}")()
				for epoch in range(epochs):
					runnable_model.train()
					#for data in train_loader:
					#	pass
					#for data in validation_loader:
					#	pass
		pass
