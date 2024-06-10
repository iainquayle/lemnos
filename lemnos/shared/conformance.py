from __future__ import annotations

from .shape import Shape

from dataclasses import dataclass

import math

@dataclass(frozen=False)
class Conformance:
	shape: Shape
	divisor: int
	def common(self, other: Conformance) -> Conformance | None:
		if (shape := self.shape.common_lossless(other.shape)) is not None:
			return Conformance(shape, math.lcm(self.divisor, other.divisor))
		else:
			return None
	def common_divisor(self, divisor: int) -> Conformance:
		return Conformance(self.shape, math.lcm(self.divisor, divisor))
	def common_shape(self, shape: Shape) -> Conformance | None:
		return self.common(Conformance(shape, 1))
