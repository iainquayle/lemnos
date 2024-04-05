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

class ExponentialGrowth:
	def __init__(self, intercept: int, exponent: float, variability: float) -> None:
		if intercept <= 0:
			raise ValueError("intercept must be greater than zero")
		if exponent <= 0:
			raise ValueError("Exponent must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._exponent: float = exponent
		self._variability: float = variability
		self._zero: int = intercept 
	def __call__(self, shape: LockedShape) -> tuple[int, int]:
		center = ((shape.get_product() / self._zero) ** self._exponent) * self._zero
		return int(center * (1 - self._variability)), int(center * (1 + self._variability))

class SchemaNode:
	__slots__ = ["_transform", "_transition_groups", "_growth_function", "_divisor_hint", "_merge_method", "debug_name", "_activation", "_regularization", "_shape_bounds"]
	def __init__(self, 
			shape_bounds: ShapeBound,
			merge_method: MergeMethod | None = None,
			transform: Transform | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			divisor_hint: int = 1,
			debug_name: str = "") -> None:
		self._shape_bounds: ShapeBound = shape_bounds 
		self._growth_function: Callable[[LockedShape], float] | None = None 
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
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, divisor: int, index: CompileIndex) -> LockedShape | None:
		if self._activation is not None:
			output_conformance = self._activation.get_conformance(output_conformance)
		growth_factor = self._growth_function(input_shape) if self._growth_function is not None else 1
		output_shape = self._transform.get_output_shape(input_shape, output_conformance, self._shape_bounds, index) if self._transform is not None else input_shape
		if output_shape is not None and output_shape in self._shape_bounds and output_conformance.compatible(output_shape): 
			return output_shape 
		else:
			return None
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		if self._merge_method is None:
			if len(input_shapes) > 1:
				raise ValueError("No merge method defined for multiple inputs")
			return OpenShape()
		else:
			return self._merge_method.get_conformance_shape(input_shapes)
	def get_conformance_divisor(self) -> int:
		if self._transform is not None:
			if (transform_divisor := self._transform.get_divisor()) is None:
				return 1
			else:
				return math.lcm(transform_divisor, self._activation.get_divisor()) if self._activation is not None else transform_divisor
		elif self._activation is not None:
			return self._activation.get_divisor()
		else:
			return 1
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
	__slots__ = ["_next", "_priority", "_join_type"]
	def __init__(self, next: SchemaNode, priority: int, join_type: JoinType = JoinType.NEW) -> None:
		if priority > MAX_PRIORITY or priority < MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self._next: SchemaNode = next
		self._priority: int = priority 
		self._join_type: JoinType = join_type 
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

