from __future__ import annotations

from ...shared import LockedShape, OpenShape, ShapeBound, ShapeConformance

import math

from abc import ABC as Abstract, abstractmethod 

class Transform(Abstract):
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		pass
	@abstractmethod
	def get_output_shape(self, input_shape: LockedShape, output_conformance: ShapeConformance, shape_bounds: ShapeBound, growth_factor: float) -> LockedShape | None:
		pass
	def get_known_divisor(self) -> int:
		return 1

class Full(Transform):
	def __init__(self) -> None:
		pass
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		return shape_in.dimensionality() == shape_out.dimensionality()
	def get_output_shape(self, input_shape: LockedShape, output_conformance: ShapeConformance, shape_bounds: ShapeBound, growth_factor: float) -> LockedShape | None:
		upper_shape = input_shape.to_open()
		if output_conformance.shape.is_locked():
			channel_raw = output_conformance.shape.get_product() // upper_shape.get_product()
			if channel_raw % output_conformance.divisor == 0:
				return upper_shape.to_locked(channel_raw) 
			return None
		else:
			channel_raw = shape_bounds.clamp_value(int(input_shape[0] * growth_factor), 0)
			if (channel_raw := _closest_divisible(channel_raw, output_conformance.divisor, shape_bounds)) is not None:
				return upper_shape.to_locked(channel_raw)
			return None


class KernelBase(Transform, Abstract):
	__slots__ = ["_kernel", "_stride", "_dilation", "_padding"]
	def __init__(self,
			kernel: tuple | int = 1, 
			padding: tuple | int = 0,
			stride: tuple | int = 1, 
			dilation: tuple | int = 1,
			) -> None:
		self._kernel: _Clamptuple = _Clamptuple(kernel)
		self._padding: _Clamptuple = _Clamptuple(padding)
		self._stride: _Clamptuple = _Clamptuple(stride)
		self._dilation: _Clamptuple = _Clamptuple(dilation)
	def output_dim_to_input_dim(self, output_shape: LockedShape, index: int) -> int:
		index -= 1 #remove the channel dimension
		return (output_shape[index + 1] - 1) * self._stride[index] + (self._kernel[index] * self._dilation[index] - (self._dilation[index] - 1)) - self._padding[index] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, index: int) -> int:
		index -= 1
		return ((input_shape[index + 1] + self._padding[index] * 2) - (self._kernel[index] * self._dilation[index] - (self._dilation[index] - 1))) // self._stride[index] + 1
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (shape_out[0] == shape_in[0])
	def get_known_divisor(self) -> int:
		return 1
	def get_kernel(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._kernel.expand(input_shape.dimensionality() - 1)
	def get_stride(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._stride.expand(input_shape.dimensionality() - 1)
	def get_dilation(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._dilation.expand(input_shape.dimensionality() - 1)
	def get_padding(self, input_shape: LockedShape) -> tuple[int, ...]:
		return self._padding.expand(input_shape.dimensionality() - 1)

class MaxPool(KernelBase):
	pass

class Conv(KernelBase):
	__slots__ = ["_kernel", "_stride", "_dilation", "_padding", "_groups"]
	def __init__(self,
			kernel: tuple | int = 1, 
			padding: tuple | int = 0,
			stride: tuple | int = 1, 
			dilation: tuple | int = 1,
			groups: int = 1,
			) -> None:
		self._kernel: _Clamptuple = _Clamptuple(kernel)
		self._padding: _Clamptuple = _Clamptuple(padding)
		self._stride: _Clamptuple = _Clamptuple(stride)
		self._dilation: _Clamptuple = _Clamptuple(dilation)
		self._groups: int = groups
	def get_output_shape(self, input_shape: LockedShape, output_conformance: ShapeConformance, shape_bounds: ShapeBound, growth_factor: float) -> LockedShape | None:
		if len(input_shape) < 2:
			raise ValueError("input shape must have at least 2 dimensions")
		upper_shape = OpenShape(*(self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))))
		if output_conformance.shape.is_locked():
			output_shape = upper_shape.to_locked(output_conformance.shape.get_product() // upper_shape.get_product())
			divisor = output_conformance.get_divisor(self._groups)
			if input_shape[0] % self._groups == 0 and output_shape[0] % divisor == 0 and shape_bounds.contains_value(output_shape[0], 0):
				return upper_shape.to_locked(output_conformance.shape.get_product() // upper_shape.get_product())
		else:
			proposed_output_shape = upper_shape.to_locked(shape_bounds.clamp_value(int(input_shape[0] * growth_factor), 0))
			divisor = output_conformance.get_divisor(self._groups)
			if input_shape[0] % self._groups == 0 and (channels := _closest_divisible(proposed_output_shape[0], divisor, shape_bounds)) is not None:
				return upper_shape.to_locked(channels)
	def get_known_divisor(self) -> int:
		return self._groups
	def get_groups(self) -> int:
		return self._groups

class Grouping(Abstract):
	@abstractmethod
	def get_proposed_groups(self, input_shape: LockedShape, proposed_output_shape: LockedShape ) -> int:
		pass
class ConstantGrouping(Grouping):
	def __init__(self, groups: int) -> None:
		if groups < 1:
			raise ValueError("groups must be greater than 0")
		self._groups: int = groups
	def get_proposed_groups(self, input_shape: LockedShape, proposed_output_shape: LockedShape ) -> int:
		return self._groups
class DepthwiseGrouping(Grouping):
	def get_proposed_groups(self, input_shape: LockedShape, proposed_output_shape: LockedShape ) -> int:
		return input_shape[0]
class InputSqrtGrouping(Grouping):
	def __init__(self, sqrt_input_scale: float = 1.0) -> None:
		self._sqrt_input_scale: float = sqrt_input_scale
	def get_proposed_groups(self, input_shape: LockedShape, proposed_output_shape: LockedShape ) -> int:
		return int(math.sqrt(input_shape[0] * self._sqrt_input_scale)) 

class FlexibleConv(KernelBase):
	__slots__ = ["_kernel", "_stride", "_dilation", "_padding", "_groups"]
	def __init__(self,
			kernel: tuple | int = 1, 
			padding: tuple | int = 0,
			stride: tuple | int = 1, 
			dilation: tuple | int = 1,
			groups: int | Grouping = 1,
			) -> None:
		self._kernel: _Clamptuple = _Clamptuple(kernel)
		self._padding: _Clamptuple = _Clamptuple(padding)
		self._stride: _Clamptuple = _Clamptuple(stride)
		self._dilation: _Clamptuple = _Clamptuple(dilation)
		self._groups: Grouping = ConstantGrouping(groups) if isinstance(groups, int) else groups 
	def get_output_shape(self, input_shape: LockedShape, output_conformance: ShapeConformance, shape_bounds: ShapeBound, growth_factor: float) -> LockedShape | None:
		if len(input_shape) < 2:
			raise ValueError("input shape must have at least 2 dimensions")
		upper_shape = OpenShape(*(self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))))
		if output_conformance.shape.is_locked():
			output_shape = upper_shape.to_locked(output_conformance.shape.get_product() // upper_shape.get_product())
			groups = self._groups.get_proposed_groups(input_shape, output_shape)
			if groups <= 1:
				raise ValueError("groups must end up being greater than 1")
			divisor = output_conformance.get_divisor(groups)
			if output_shape[0] % divisor == 0 and shape_bounds.contains_value(output_shape[0], 0):
				return upper_shape.to_locked(output_conformance.shape.get_product() // upper_shape.get_product())
		else:
			proposed_output_shape = upper_shape.to_locked(shape_bounds.clamp_value(int(input_shape[0] * growth_factor), 0))
			groups = self._groups.get_proposed_groups(input_shape, proposed_output_shape)
			if groups <= 1:
				raise ValueError("groups must end up being greater than 1")
			divisor = output_conformance.get_divisor(groups)
			if (channels := _closest_divisible(proposed_output_shape[0], divisor, shape_bounds)) is not None:
				return upper_shape.to_locked(channels)
	def get_known_divisor(self) -> int:
		return 1
	def get_internal_splits(self, input_shape: LockedShape, output_shape: LockedShape) -> list[tuple[int, int, int]]:
		groups = self._groups.get_proposed_groups(input_shape, output_shape)
		base_group_size_in = input_shape[0] // groups
		extra_channels_in = input_shape[0] % base_group_size_in
		group_infos_in = [(groups - extra_channels_in, base_group_size_in), (extra_channels_in, base_group_size_in + 1)]
		base_group_size_out = output_shape[0] // groups
		extra_channels_out = output_shape[0] % base_group_size_out
		group_infos_out = [(groups - extra_channels_out, base_group_size_out), (extra_channels_out, base_group_size_out + 1)]
		in_groups_greater = group_infos_in[0][0] > group_infos_out[0][0]
		groups_infos = [(min(group_infos_in[0][0], group_infos_out[0][0]), group_infos_in[0][1], group_infos_out[0][1]),
			(abs(group_infos_in[0][0] - group_infos_out[0][0]), group_infos_in[0 if in_groups_greater else 1][1], group_infos_out[1 if in_groups_greater else 0][1]),
			(min(group_infos_in[1][0], group_infos_out[1][0]), group_infos_in[1][1], group_infos_out[1][1])]
		return [(group_size_in * groups, group_size_out * groups, groups) for groups, group_size_in, group_size_out in groups_infos]

def _closest_divisible(value: int, divisor: int, shape_bound: ShapeBound) -> int | None:
	lower, upper = _closest_divisibles(value, divisor)
	if shape_bound.contains_value(lower, 0) and (abs(value - lower) < abs(value - upper) or not shape_bound.contains_value(upper, 0)):
		return lower
	elif shape_bound.contains_value(upper, 0):
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
