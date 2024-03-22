from __future__ import annotations

from ...shared import Shape, LockedShape, OpenShape, ShapeBound 
from ..ir_index import IRIndex
from ...target import TargetComponents


from abc import ABC as Abstract, abstractmethod 

_LOWER = 0
_UPPER = 1

class Transform(Abstract):
	__slots__ = ["_size_coeffs_bounds"]
	def __init__(self, size_coeffs_bounds: float | tuple[float, float]) -> None:
		if isinstance(size_coeffs_bounds, float):
			size_coeffs_bounds = (size_coeffs_bounds, size_coeffs_bounds)
		elif isinstance(size_coeffs_bounds, int):
			raise ValueError("wtf")
		elif size_coeffs_bounds[0] > size_coeffs_bounds[1]:
			size_coeffs_bounds = (size_coeffs_bounds[1], size_coeffs_bounds[0])
		self._size_coeffs_bounds: tuple[float, float] = size_coeffs_bounds
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		pass
	@abstractmethod
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: IRIndex) -> LockedShape | None:
		pass
	@abstractmethod
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape, shape_out: LockedShape) -> str:	
		pass
	def get_coeff_bounds(self, size: int) -> tuple[int, int]:
		lower = int(self._size_coeffs_bounds[_LOWER] * size)
		upper = int(self._size_coeffs_bounds[_UPPER] * size)
		return (lower, upper)

class Full(Transform):
	def __init__(self, size_coeffs_bounds: float | tuple[float, float]) -> None:
		super().__init__(size_coeffs_bounds)
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		return shape_in.dimensionality() == shape_out.dimensionality()
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: IRIndex) -> LockedShape | None:
		upper_shape = input_shape.to_open()
		if output_conformance.is_locked():
			return upper_shape.to_locked(output_conformance.get_product() // upper_shape.get_product())
		else:
			return upper_shape.to_locked(shape_bounds.clamp_value(index.get_shuffled(self.get_coeff_bounds(input_shape[0]), 0), 0))
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape, shape_out: LockedShape) -> str:
		return target.full(shape_in, shape_out)

class Conv(Transform):
	__slots__ = ["_kernel", "_stride", "_dilation", "_padding", "_group_size"]
	def __init__(self,
			size_coeffs_bounds: float | tuple[float, float],
			kernel: tuple | int = 1, 
			stride: tuple | int = 1, 
			dilation: tuple | int = 1,
			padding: tuple | int = 0,
			group_size: int | None = None, 
			) -> None:
		super().__init__(size_coeffs_bounds)
		self._kernel: _Clamptuple = _Clamptuple(kernel)
		self._stride: _Clamptuple = _Clamptuple(stride)
		self._dilation: _Clamptuple = _Clamptuple(dilation)
		self._padding: _Clamptuple = _Clamptuple(padding)
		self._group_size: int | None = group_size 
	def output_dim_to_input_dim(self, output_shape: LockedShape, i: int) -> int:
		i -= 1
		return (output_shape[i + 1] - 1) * self._stride[i] + (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1)) - self._padding[i] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, i: int) -> int:
		i -= 1
		return ((input_shape[i + 1] + self._padding[i] * 2) - (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1))) // self._stride[i] + 1
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: IRIndex) -> LockedShape | None:
		if len(input_shape) < 2:
			raise ValueError("input shape must have at least 2 dimensions")
		upper_shape = OpenShape(*(self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))))
		if output_conformance.is_locked():
			channels = output_conformance.get_product() // upper_shape.get_product()
			if self._group_size is not None and (input_shape[0] % self._group_size != 0 or channels % self._group_size != 0):
				#keep the input shape check separate incase the grouping problem is fixed using padding
				return None
			else:
				return upper_shape.to_locked(channels)
		else:
			if self._group_size is not None and input_shape[0] % self._group_size != 0:
				return None	
			channels_raw = shape_bounds.clamp_value(index.get_shuffled(self.get_coeff_bounds(input_shape[0]), 0), 0)
			if self._group_size is None:
				return upper_shape.to_locked(channels_raw)
			else:
				#this aint right...
				output_shape = None
				channels_factor = input_shape[0] // self._group_size
				if shape_bounds.contains_value(channels := channels_raw // channels_factor * channels_factor, 0):
					output_shape = upper_shape.to_locked(channels)
				if (output_shape is None 
						or (shape_bounds.contains_value(channels := (channels_raw // channels_factor + 1) * channels_factor, 0)
						and channels - channels_raw < channels_raw - output_shape[0])):
					output_shape = upper_shape.to_locked(channels)
				return output_shape
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (shape_out[0] == shape_in[0])
	def get_init_src(self, target: TargetComponents, shape_in: LockedShape, shape_out: LockedShape) -> str:
		dimensionality = len(shape_in) - 1
		return target.conv(shape_in, shape_out, self._kernel.expand(dimensionality), self._stride.expand(dimensionality), self._padding.expand(dimensionality), 1 if self._group_size is None else shape_out[0] // self._group_size)

class _Clamptuple:
	slot = ["_val"]
	def __init__(self, val: tuple[int, ...] | int) -> None:
		self._val: tuple[int, ...] = val if isinstance(val, tuple) else (val,)
		if len(self._val) == 0:
			raise ValueError("empty tuple")
	def __getitem__(self, index: int) -> int:
		if index > len(self._val):
			return self._val[-1]
		else:
			return self._val[index]
	def expand(self, dimensionality: int) -> tuple[int, ...]:
		return self._val + (self._val[-1],) * (dimensionality - len(self._val))
