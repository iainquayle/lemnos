import unittest

from src.shared.merge_method import Add, Concat
from src.shared.shape import LockedShape, OpenShape 

class TestMergeMethod(unittest.TestCase):
	def setUp(self) -> None:
		self.a = LockedShape.new(2, 3)
		self.b = OpenShape.new(3)
		self.c = OpenShape.new(2, 3)
		self.d = LockedShape.new(1, 2, 3)
		self.e = LockedShape.new(4, 3)
		self.f = LockedShape.new(6)
	def test_concat_get_output(self) -> None:
		self.assertEqual(Concat().get_output_shape([self.d, self.a], 2), self.e)
	def test_add_get_output(self) -> None:
		self.assertEqual(Add().get_output_shape([self.d, self.a], 2), self.a)
	def test_concat_get_conformance(self) -> None:
		self.assertEqual(Concat().get_conformance_shape([self.d, self.a], 2), self.b)
		self.assertEqual(Concat().get_conformance_shape([], 2), OpenShape.new())
	def test_add_get_conformance(self) -> None:
		self.assertEqual(Add().get_conformance_shape([self.d, self.a], 2), self.a)
		self.assertEqual(Add().get_conformance_shape([], 2), OpenShape.new())
