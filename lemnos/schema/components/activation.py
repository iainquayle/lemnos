from __future__ import annotations

from ...shared import LockedShape, Shape, ShapeBound

from abc import ABC as Abstract 

class Activation(Abstract):
	def get_conformance(self, conformance: Shape) -> Shape:
		return conformance
	def get_output_shape(self, input_shape: LockedShape) -> LockedShape:
		return input_shape
	def get_divisor(self, divisor: int) -> int:
		return divisor 
	def get_bounds(self, bounds: ShapeBound) -> ShapeBound:
		return bounds
	def get_growth_factor(self, factor: float) -> float:
		return factor
	def scale_output_shape(self, output_shape: LockedShape) -> LockedShape:
		return output_shape
	def scale_build_conformances(self, conformance_shape: Shape, bounds: ShapeBound, divisor: int, growth_factor: float) -> tuple[Shape, ShapeBound, int, float]:
		return self.get_conformance(conformance_shape), self.get_bounds(bounds), self.get_divisor(divisor), self.get_growth_factor(growth_factor)

class GLU(Activation):
	def get_conformance(self, conformance: Shape) -> Shape:
		if conformance.is_locked():
			return conformance
		else:
			return conformance.to_open().to_locked(conformance[0] * 2)
	def get_output_shape(self, input_shape: LockedShape) -> LockedShape:
		#worried about float imprecision so not using scale, should be fine for 2s though
		return input_shape.to_open().to_locked(input_shape[0] // 2)
	def get_divisor(self, divisor: int) -> int:
		return divisor * 2 
	def get_bounds(self, bounds: ShapeBound) -> ShapeBound:
		return bounds.scale([2])
	def scale_output_shape(self, output_shape: LockedShape) -> LockedShape:
		return output_shape.to_open().to_locked(output_shape[0] * 2)
	def get_growth_factor(self, factor: float) -> float:
		return factor * 2
class ReLU(Activation):
	pass
class ReLU6(Activation):
	pass
class Softmax(Activation):
	pass
class Sigmoid(Activation):
	pass
class SiLU(Activation):
	pass
