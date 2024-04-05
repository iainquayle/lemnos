from __future__ import annotations

from ...shared import Shape, LockedShape, OpenShape, ShapeBound 

import math

from abc import ABC as Abstract, abstractmethod 
from enum import Enum

class Transform(Abstract):
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		pass
	@abstractmethod
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, divisor: int, growth_factor: float) -> LockedShape | None:
		pass
	def get_divisor(self) -> int | None:
		return None

class Full(Transform):
	def __init__(self) -> None:
		pass
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		return shape_in.dimensionality() == shape_out.dimensionality()
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, divisor: int, growth_factor: float) -> LockedShape | None:
		upper_shape = input_shape.to_open()
		if output_conformance.is_locked():
			channel_raw = output_conformance.get_product() // upper_shape.get_product()
			if channel_raw % divisor == 0:
				return upper_shape.to_locked(channel_raw) 
			return None
		else:
			if (channel_raw := _closest_divisible(int(input_shape[0] * growth_factor), divisor, shape_bounds)) is not None:
				return upper_shape.to_locked(channel_raw)
class GroupType(Enum):
	DEPTHWISE = "depthwise"
class Conv(Transform):
	__slots__ = ["_kernel", "_stride", "_dilation", "_padding", "_groups"]
	def __init__(self,
			kernel: tuple | int = 1, 
			stride: tuple | int = 1, 
			dilation: tuple | int = 1,
			padding: tuple | int = 0,
			groups: int | GroupType = 1, #none would be depthwise, so needs to be switched to some other signifier than None
			) -> None:
		self._kernel: _Clamptuple = _Clamptuple(kernel)
		self._stride: _Clamptuple = _Clamptuple(stride)
		self._dilation: _Clamptuple = _Clamptuple(dilation)
		self._padding: _Clamptuple = _Clamptuple(padding)
		if isinstance(groups, int) and groups < 1:
			raise ValueError("groups must be greater than 0")
		self._groups: int | GroupType = groups
	def output_dim_to_input_dim(self, output_shape: LockedShape, i: int) -> int:
		i -= 1
		return (output_shape[i + 1] - 1) * self._stride[i] + (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1)) - self._padding[i] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, i: int) -> int:
		i -= 1
		return ((input_shape[i + 1] + self._padding[i] * 2) - (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1))) // self._stride[i] + 1
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, divisor: int, growth_factor: float) -> LockedShape | None:
		if len(input_shape) < 2:
			raise ValueError("input shape must have at least 2 dimensions")
		groups: int = 1 
		if isinstance(self._groups, int):
			divisor = math.lcm(divisor, self._groups)
			groups = self._groups
		elif self._groups == GroupType.DEPTHWISE:
			divisor = math.lcm(divisor, input_shape[0])
			groups = input_shape[0]
		else:
			raise NotImplementedError("group type not supported yet")
		upper_shape = OpenShape(*(self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))))
		if output_conformance.is_locked():
			channels = output_conformance.get_product() // upper_shape.get_product()
			if input_shape[0] % groups == 0 and channels % divisor == 0 and shape_bounds.contains_value(channels, 0):
				return upper_shape.to_locked(channels)
			return None
		else:
			if input_shape[0] % groups != 0:
				return None	
			channels_raw = shape_bounds.clamp_value(int(input_shape[0] * growth_factor), 0)
			if (channels := _closest_divisible(channels_raw, divisor, shape_bounds)) is not None:
				return upper_shape.to_locked(channels)
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (shape_out[0] == shape_in[0])
	def get_divisor(self) -> int:
		if isinstance(self._groups, int):
			return self._groups
		elif self._groups == GroupType.DEPTHWISE:
			return 1
		else:
			raise NotImplementedError("group type not supported yet")
	def get_kernel(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._kernel.expand(input_shape.dimensionality() - 1)
	def get_stride(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._stride.expand(input_shape.dimensionality() - 1)
	def get_dilation(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._dilation.expand(input_shape.dimensionality() - 1)
	def get_padding(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._padding.expand(input_shape.dimensionality() - 1)
	def get_groups(self) -> int | GroupType:
		return self._groups

def _closest_divisible(value: int, divisor: int, shape_bound: ShapeBound) -> int | None:
	lower, upper = _closest_divisibles(value, divisor)
	if shape_bound.contains_value(lower, 0) and abs(value - lower) < abs(value - upper):
		return lower
	elif shape_bound.contains_value(upper, 0) and abs(value - upper) < abs(value - lower):
		return upper
	else:
		return None 
def _closest_divisibles(value: int, divisor: int) -> tuple[int, int]:
	lower = value // divisor * divisor
	upper = lower + divisor
	return lower, upper
class _Clamptuple:
	slot = ["_val"]
	def __init__(self, val: tuple[int, ...] | int) -> None:
		self._val: tuple[int, ...] = val if isinstance(val, tuple) else (val,)
		if len(self._val) == 0:
			raise ValueError("empty tuple")
	def __getitem__(self, index: int) -> int:
		if index >= len(self._val):
			return self._val[-1]
		else:
			return self._val[index]
	def expand(self, dimensionality: int) -> tuple[int, ...]:
		return self._val + (self._val[-1],) * (dimensionality - len(self._val))
