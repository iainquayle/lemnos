from __future__ import annotations

from src.model import Model
from src.schema import Schema

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module

class Control:
	def __init__(self, schema: Schema, train_dataset: Dataset, validation_dataset: Dataset) -> None:
		self._schema: Schema = schema
		self._train_dataset: Dataset = train_dataset
		self._validation_dataset: Dataset = validation_dataset
	def optimize(self, save_dir: str, gene_pool_size: int, optimizer: Optimizer, criterion: Module) -> None:
		#look into taking in a sampler for the data loader
		#could take in a list of optimizers and criterions to all test
		train_loader = DataLoader(self._train_dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=True, pin_memory=True)
		validation_loader = DataLoader(self._validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
		pass
