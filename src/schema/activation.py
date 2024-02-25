from __future__ import annotations

from .src_generation import * 
from src.shared import LockedShape

from abc import ABC as Abstract, abstractmethod

class Activation(Abstract):
	@abstractmethod
	def get_init_src(self, shape_in: LockedShape) -> str:
		pass


class ReLU(Activation):
	def get_init_src(self, shape_in: LockedShape) -> str:
		return relu_()
class ReLU6(Activation):
	def get_init_src(self, shape_in: LockedShape) -> str:
		return relu6_() 
