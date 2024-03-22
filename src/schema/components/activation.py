from __future__ import annotations

from ...shared import LockedShape
from ...target.target_components import TargetComponents
from .component import Component

from abc import ABC as Abstract 

class Activation(Component, Abstract):
	pass

class ReLU(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.relu_init()]
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
class ReLU6(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.relu6_init()] 
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
class Softmax(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.softmax_init()] 
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
class Sigmoid(Activation):
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [target.sigmoid_init()] 
	def get_forward_src(self, target: TargetComponents, input_expr: str, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		return [input_expr] 
