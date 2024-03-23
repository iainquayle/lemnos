from __future__ import annotations

from .transform import Transform
from .activation import Activation 
from .regularization import Regularization
from .merge_method import MergeMethod

Component = Transform | Activation | Regularization | MergeMethod
