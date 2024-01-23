from abc import ABC as Abstract, abstractmethod

from typing import Any
from typing import Callable, TypeVar 

#interface for simplified reverse indexing, may just be overkill
#from looking at the @dataclass decorator, it seems that it would be possible to make a more effective decorator that will give correct typing
class RevereseGet(Abstract):
	@abstractmethod
	def __getitem__(self, index: int) -> Any:
		pass
	@abstractmethod
	def __len__(self) -> int:
		pass
	@abstractmethod
	def reverse_get(self, index: int) -> Any:
		pass

#make empty function with correct typing, and then replace it with the actual function
T = TypeVar("T")
def reverse_func(_: Callable[[RevereseGet, int], T]) -> Callable[[RevereseGet, int], T]:
	def replace(self, index: int) -> T:
		return self[len(self) - index - 1]
	return replace
