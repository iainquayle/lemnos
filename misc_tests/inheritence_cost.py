import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

class A:
	__slots__ = ["_flag", "_aaaaaaaaaaaa"]
	def __init__(self) -> None:
		self._flag = random.choice([True, False])
		self._aaaaaaaaaaaa = random.randint(0, 100)
	def access(self, i):
		return self.action(i)
	def action(self, i):
		return self.second_action(self._aaaaaaaaaaaa + (i if self._flag else -i))
	def second_action(self, i):
		return self._aaaaaaaaaaaa * (i if self._flag else -i)
	

class S(ABC):
	__slots__ = ["_aaaaaaaaaaaa"]
	def __init__(self) -> None:
		self._aaaaaaaaaaaa: int = random.randint(0, 100)
	@abstractmethod	
	def action(self, i):
		pass
	@abstractmethod	
	def second_action(self, i):
		pass
	def access(self, i):
		return self.action(i)

class AS(S):
	def action(self, i):
		return self.second_action(self._aaaaaaaaaaaa + i)
	def second_action(self, i):
		return self._aaaaaaaaaaaa * i

class BS(S):
	def action(self, i):
		return self.second_action(self._aaaaaaaaaaaa - i)
	def second_action(self, i):
		return self._aaaaaaaaaaaa * -i


NUM_CLASSES = 10000
no_inherit = [A() for _ in range(NUM_CLASSES)]
inherit = [(AS() if i._flag else BS()) for i in no_inherit]

NUM_ITERS = 1000 

def test_no_inheritence():
	for i in range(NUM_ITERS):
		for c in no_inherit:
			c.access(i)

def test_inheritence():
	for i in range(NUM_ITERS):
		for c in inherit:
			c.access(i)

def time_func(func):
	start = time.time()
	func()
	return time.time() - start

print("none", time_func(test_no_inheritence))
print("with", time_func(test_inheritence))
print("none", time_func(test_no_inheritence))
print("with", time_func(test_inheritence))


