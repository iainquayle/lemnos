from __future__ import annotations

from ..shared import LockedShape, Shape, ShapeBound
from .components import Activation, Regularization, Transform, MergeMethod
from .ir_index import IRIndex

from typing import Iterator, Iterable 
from typing_extensions import Self

from enum import Enum

class SchemaNode:
	__slots__ = ["_transform", "_transition_groups", "_merge_method", "debug_name", "_activation", "_regularization", "_shape_bounds"]
	def __init__(self, 
			shape_bounds: ShapeBound,
			merge_method: MergeMethod,
			transform: Transform | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			debug_name: str = "") -> None:
		self._shape_bounds: ShapeBound = shape_bounds 
		self._transition_groups: list[TransitionGroup] = []
		self._merge_method: MergeMethod = merge_method 
		self._transform: Transform | None = transform 
		self._activation: Activation | None = activation 
		self._regularization: Regularization | None = regularization 
		self.debug_name: str = debug_name 
	def add_group(self, *transitions: tuple[SchemaNode, int, JoinType] | Transition) -> Self:
		self._transition_groups.append(TransitionGroup([transition if isinstance(transition, Transition) else Transition(*transition) for transition in transitions]))
		return self
	def get_input_shape(self, input_shapes: list[LockedShape]) -> LockedShape:
		return self._merge_method.get_merged_shape(input_shapes).squash(self.dimensionality())
	def get_output_shape(self, mould_shape: LockedShape, output_conformance: Shape, index: IRIndex) -> LockedShape | None:
		output_shape = self._transform.get_output_shape(mould_shape, output_conformance, self._shape_bounds, index) if self._transform is not None else mould_shape
		if output_shape is not None and output_shape in self._shape_bounds and output_conformance.compatible(output_shape): 
			return output_shape 
		else:
			return None
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		return self._merge_method.get_conformance_shape(input_shapes)
	def get_transform(self) -> Transform | None:
		return self._transform
	def get_merge_method(self) -> MergeMethod:
		return self._merge_method
	def dimensionality(self) -> int:
		return len(self._shape_bounds)
	def __getitem__(self, index: int) -> TransitionGroup:
		return self._transition_groups[index]
	def __iter__(self) -> Iterator[TransitionGroup]:
		return iter(self._transition_groups)
	def __len__(self) -> int:
		return len(self._transition_groups)
	def get_inits_src(self, mould_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		src: list[str] = []
		if self._transform is not None:
			src.append(self._transform.get_init_src(mould_shape, output_shape))
		if self._activation is not None:
			src.append(self._activation.get_init_src(mould_shape))
		if self._regularization is not None:
			src.append(self._regularization.get_init_src(mould_shape))
		return src


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

