from __future__ import annotations

from .transforms import Transform
from .activations import Activation 
from .regularizations import Regularization
from .merge_method import MergeMethod

from abc import ABC as Abstract

Component = Transform | Activation | Regularization | MergeMethod

class _Component(Abstract):
	pass
