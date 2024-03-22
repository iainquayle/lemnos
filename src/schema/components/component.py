from __future__ import annotations

from ...shared import LockedShape
from ...target.target_components import TargetComponents

from abc import ABC as Abstract, abstractmethod

class Component(Abstract):
	@abstractmethod
	def get_inits_src(self, target: TargetComponents, input_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		pass
