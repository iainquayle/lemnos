import unittest

from torch import nn, Size

from src.build_structures.node_parameters import NodeParameters, IdentityInfo
from src.build_structures.commons import Bound, Index, Concat, Add

class TestMergeMethod(unittest.TestCase):
	def setUp(self) -> None:
		self.base_shape = Size([3, 32])
		self.none_valid = self.base_shape + Size([1, 32])
		self.none_valid = self.base_shape + Size([1, 0])
	def test_concat(self) -> None:
		pass

class TestIdentityInfo(unittest.TestCase):
	def setUp(self) -> None:
		pass
	def test_output_shape(self) -> None:
		pass
	def test_output_shape_fail(self) -> None:
		pass
