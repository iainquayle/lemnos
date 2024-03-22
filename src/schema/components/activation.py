from __future__ import annotations

from ...shared import LockedShape
from ...target import TargetComponents

from abc import ABC as Abstract, abstractmethod

class Activation(Abstract):
	@abstractmethod
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		pass

class ReLU(Activation):
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.relu()
class ReLU6(Activation):
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.relu6() 
class Softmax(Activation):
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.softmax()
class Sigmoid(Activation):
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape) -> str:
		return target.sigmoid()
