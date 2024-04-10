from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, ShapeBound
from .components.transform import Transform
from .components.activation import Activation
from .components.regularization import Regularization
from .components.merge_method import MergeMethod
from .components.component import Component
from .compile_index import CompileIndex 

import math

from typing import Iterator, Iterable, Callable 
from typing_extensions import Self

from enum import Enum
from dataclasses import dataclass


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

class SchemaNode:
	__slots__ = ["_transform", "_transition_groups", "_growth_function", "_divisor_hint", "_merge_method", "debug_name", "_activation", "_regularization", "_shape_bounds"]
	def __init__(self, 
			shape_bounds: ShapeBound,
			growth_function: Callable[[LockedShape, CompileIndex], float] | None = None,
			merge_method: MergeMethod | None = None,
			transform: Transform | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			divisor_hint: int = 1,
			debug_name: str = "") -> None:
		self._shape_bounds: ShapeBound = shape_bounds 
		self._growth_function: Callable[[LockedShape, CompileIndex], float] | None = growth_function 
		self._transition_groups: list[TransitionGroup] = []
		self._merge_method: MergeMethod | None = merge_method 
		self._transform: Transform | None = transform 
		self._activation: Activation | None = activation 
		self._regularization: Regularization | None = regularization 
		self._divisor_hint: int = divisor_hint 
		self.debug_name: str = debug_name 
	def add_group(self, *transitions: tuple[SchemaNode, int, JoinType] | Transition) -> Self:
		self._transition_groups.append(TransitionGroup([transition if isinstance(transition, Transition) else Transition(*transition) for transition in transitions]))
		return self
	def get_input_shape(self, input_shapes: list[LockedShape]) -> LockedShape:
		if self._merge_method is None:
			if len(input_shapes) > 1:
				raise ValueError("No merge method defined for multiple inputs")
			return input_shapes[0].squash(self.dimensionality())
		else:
			return self._merge_method.get_merged_shape(input_shapes).squash(self.dimensionality())
	def get_output_shape(self, input_shape: LockedShape, conformance: Conformance, index: CompileIndex) -> LockedShape | None:
		conformance_divisor = math.lcm(conformance.divisor, self._divisor_hint)
		growth_factor = self._growth_function(input_shape, index) if self._growth_function is not None else 1#index.get_shuffled((,), 0)
		conformance_shape = conformance.shape
		bounds = self._shape_bounds
		if self._activation is not None:
			conformance_shape, bounds, conformance_divisor, growth_factor = self._activation.scale_build_conformances(conformance_shape, bounds, conformance_divisor, growth_factor)
		output_shape = self._transform.get_output_shape(input_shape, conformance_shape, bounds, conformance_divisor, growth_factor) if self._transform is not None else input_shape
		if output_shape is not None:
			output_shape = self._activation.scale_output_shape(output_shape) if self._activation is not None else output_shape
			if output_shape in self._shape_bounds and conformance_shape.compatible(output_shape): 
				return output_shape 
		else:
			return None
	def get_conformance(self, sibling_shapes: list[LockedShape]) -> Conformance | None:
		if self._merge_method is None:
			if len(sibling_shapes) > 1:
				raise ValueError("No merge method defined for multiple inputs")
			return Conformance(OpenShape(), self._divisor_hint)
		elif (conformance_shape := self._merge_method.get_conformance_shape(sibling_shapes)) is not None:
			divisor = math.lcm(self._divisor_hint, self._transform.get_divisor()) if self._transform is not None else self._divisor_hint 
			return Conformance(conformance_shape, self._activation.get_divisor(divisor) if self._activation is not None else divisor)
	def get_transform(self) -> Transform | None:
		return self._transform
	def get_merge_method(self) -> MergeMethod | None:
		return self._merge_method
	def get_components(self) -> list[Component]:
		return [component for component in (self._merge_method, self._transform, self._activation, self._regularization) if component is not None]
	def dimensionality(self) -> int:
		return len(self._shape_bounds)
	def __getitem__(self, index: int) -> TransitionGroup:
		return self._transition_groups[index]
	def __iter__(self) -> Iterator[TransitionGroup]:
		return iter(self._transition_groups)
	def __len__(self) -> int:
		return len(self._transition_groups)

class JoinType(Enum):
	EXISTING = "existing"
	NEW = "new"
	AUTO = "auto"

MAX_PRIORITY: int = 128 
MIN_PRIORITY: int = 0 
class Transition:
	__slots__ = ["_next", "_priority", "_join_type", "_growth_function"]
	def __init__(self, next: SchemaNode, priority: int, join_type: JoinType = JoinType.NEW) -> None:
		if priority > MAX_PRIORITY or priority < MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self._next: SchemaNode = next
		self._priority: int = priority 
		self._join_type: JoinType = join_type 
		self._growth_function: Callable[[LockedShape, CompileIndex], float] | None = None
	def get_next(self) -> SchemaNode:
		return self._next
	def get_priority(self) -> int:
		return self._priority
	def get_join_type(self) -> JoinType:
		return self._join_type

class TransitionGroup:
	__slots__ = ["_transitions"]
	def __init__(self, transitions: Iterable[Transition]) -> None:
		pattern_set: set[SchemaNode] = set()
		for transition in transitions:
			if transition.get_next() in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.get_next())
		self._transitions: tuple[Transition, ...] = tuple(transitions) 
	def __iter__(self) -> Iterator[Transition]:
		return iter(self._transitions)
	def __len__(self) -> int:
		return len(self._transitions)

