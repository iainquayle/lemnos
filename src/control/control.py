from __future__ import annotations

from src.model import Model
from src.schema import Schema

class Control:
	def __init__(self, schema: Schema):
		self._schema = schema
	def optimize(self, save_dir: str, pool_size: int) -> None:
		#needed:
		#	training and validation data
		#	optimizer
		#	loss scheme
		#	hyper parameters dealing with search
		pass
