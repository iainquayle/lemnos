from __future__ import annotations

from ...shared import LockedShape
from ...target.target_components import TargetComponents
from .component import Component

from abc import ABC as Abstract 

class Activation(Component, Abstract):
	pass

class ReLU(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.relu()]
class ReLU6(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.relu6()] 
class Softmax(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.softmax()] 
class Sigmoid(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.sigmoid()] 
