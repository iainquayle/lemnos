from __future__ import annotations

from typing import Callable
from abc import ABC as Abstract, abstractmethod

#may not be best way, as how would a user be able to extend to use a new module
#backends could only be relied upon to have functions set out in this class
#may be best to just start this way, and change if necessary
#other option, have a mapping of components to functions, and when generating, it picks the function from the mapping
#	slightly jank and lose some typing, but very extensible
#src_funcs: dict[Transform | Activation | Regularization, Callable] = {}

class SrcBackend(Abstract):
	@abstractmethod
	def module() -> str:
		pass
	pass
