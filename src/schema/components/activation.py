from __future__ import annotations

from ...shared import Shape, ShapeBound

from abc import ABC as Abstract 

class Activation(Abstract):
	def get_conformance(self, conformance: Shape) -> Shape:
		return conformance
	def get_divisor(self) -> int:
		return 1 
	def get_bounds(self, bounds: ShapeBound) -> ShapeBound:
		return bounds

class GLU(Activation):
	def get_conformance(self, conformance: Shape) -> Shape:
		if conformance.is_locked():
			return conformance
		else:
			return conformance.to_open().to_locked(conformance[0] * 2)
	def get_divisor(self) -> int:
		return 2
	def get_bounds(self, bounds: ShapeBound) -> ShapeBound:
		listed_bounds = bounds.get_bounds() 
		a, b = listed_bounds[0]
		if a is not None:
			a *= 2
		if b is not None:
			b *= 2
		listed_bounds[0] = (a, b)
		return ShapeBound() 
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
