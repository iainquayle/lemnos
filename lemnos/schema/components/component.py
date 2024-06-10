from __future__ import annotations

from .transform import Transform
from .activation import Activation 
from .regularization import Regularization
from .merge_method import MergeMethod

from abc import ABC as Abstract

Component = Transform | Activation | Regularization | MergeMethod

class _Component(Abstract):
	pass
