from __future__ import annotations

from .src_generation import *
from src.shared import LockedShape

from abc import ABC as Abstract, abstractmethod

class Regularization(Abstract):
	@abstractmethod	
	def get_init_src(self, shape_in: LockedShape) -> str:
		pass

class Dropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, shape_in: LockedShape) -> str:
		return dropout_(self._p)
class ChannelDropout(Regularization):
	def __init__(self, p: float) -> None:
		self._p: float = p
	def get_init_src(self, shape_in: LockedShape) -> str:
		return channel_dropout_(self._p, shape_in)
class BatchNormalization(Regularization):
	def get_init_src(self, shape_in: LockedShape) -> str:
		return batch_norm_(shape_in)
