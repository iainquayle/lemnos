from __future__ import annotations

from src.shared import LockedShape, Shape, Index, ShapeBound
from .merge_method import MergeMethod 
from .activation import Activation
from .regularization import Regularization
from .transform import Transform  

from typing import List, Set, Iterable, Tuple
from typing_extensions import Self
from copy import copy 

class SchemaNode:
	__slots__ = ["_transform", "_transition_groups", "_merge_method", "debug_name", "_activation", "_regularization", "_shape_bounds"]
	def __init__(self, shape_bounds: ShapeBound,
			merge_method: MergeMethod,
			transform: Transform | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			debug_name: str = "") -> None:
		self._shape_bounds: ShapeBound = shape_bounds 
		self._transition_groups: List[TransitionGroup] = []
		self._merge_method: MergeMethod = merge_method 
		self._transform: Transform | None = transform 
		self._activation: Activation | None = activation 
		self._regularization: Regularization | None = regularization 
		self.debug_name: str = debug_name 
	def add_group(self, repetition_bounds: ShapeBound, *transitions: Tuple[SchemaNode, int, bool] | Transition) -> Self:
		self._transition_groups.append(TransitionGroup(repetition_bounds, [transition if isinstance(transition, Transition) else Transition(*transition) for transition in transitions]))
		return self
	def get_mould_shape(self, input_shapes: List[LockedShape]) -> LockedShape:
		return self._merge_method.get_output_shape(input_shapes).squash(self.dimensionality())
	def get_output_shape(self, mould_shape: LockedShape, output_conformance: Shape, index: Index) -> LockedShape | None:
		output_shape = self._transform.get_output_shape(mould_shape, output_conformance, self._shape_bounds, index) if self._transform is not None else mould_shape
		#if not isinstance(output_shape, LockedShape):
		#	raise ValueError("output shape is not locked shape")
		if output_shape is not None and output_shape in self._shape_bounds and output_conformance.compatible(output_shape): 
			return output_shape 
		else:
			return None
	def get_conformance_shape(self, input_shapes: List[LockedShape]) -> Shape:
		return self._merge_method.get_conformance_shape(input_shapes)
	def get_transform(self) -> Transform | None:
		return self._transform
	def get_merge_method(self) -> MergeMethod:
		return self._merge_method
	def get_transition_groups(self) -> List[TransitionGroup]:
		return self._transition_groups
	def dimensionality(self) -> int:
		return len(self._shape_bounds)
	def __getitem__(self, index: int) -> TransitionGroup:
		return self._transition_groups[index]
	def __iter__(self) -> Iterable[TransitionGroup]:
		return iter(self._transition_groups)
	def get_inits_src(self, mould_shape: LockedShape, output_shape: LockedShape) -> List[str]:
		src: List[str] = []
		if self._transform is not None:
			src.append(self._transform.get_init_src(mould_shape, output_shape))
		if self._activation is not None:
			src.append(self._activation.get_init_src(mould_shape))
		if self._regularization is not None:
			src.append(self._regularization.get_init_src(mould_shape))
		return src

class Transition:
	_MAX_PRIORITY: int = 128 
	_MIN_PRIORITY: int = 0 
	__slots__ = ["_next", "_optional", "_priority", "_join_existing"]
	def __init__(self, next: SchemaNode, priority: int, join_existing: bool = False) -> None:
		if priority > Transition._MAX_PRIORITY or priority < Transition._MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self._next: SchemaNode = next
		self._optional: bool =  False 
		self._priority: int = priority 
		self._join_existing: bool = join_existing 
	def get_next(self) -> SchemaNode:
		return self._next
	def get_priority(self) -> int:
		return self._priority
	#def is_optional(self) -> bool:
	#	return self._optional
	def get_join_existing(self) -> bool:
		return self._join_existing
	@staticmethod
	def get_max_priority() -> int:
		return Transition._MAX_PRIORITY 
	@staticmethod
	def get_min_priority() -> int:
		return Transition._MIN_PRIORITY

class TransitionGroup:
	__slots__ = ["_transitions", "_repetition_bounds"]
	def __init__(self, repetition_bounds: ShapeBound, transitions: List[Transition]) -> None:
		self._transitions: List[Transition] = copy(transitions)
		self._repetition_bounds: ShapeBound = repetition_bounds
	def set_transitions(self, transitions: List[Transition]) -> None:
		pattern_set: Set[SchemaNode] = set()
		for transition in transitions:
			if transition.get_next() in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.get_next())
		self._transitions = transitions
	def get_bounds(self) -> ShapeBound:
		return self._repetition_bounds
	def __getitem__(self, index: int) -> Transition:
		return self._transitions[index]
	def __iter__(self) -> Iterable[Transition]:
		return iter(self._transitions)
	def __len__(self) -> int:
		return len(self._transitions)
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)
