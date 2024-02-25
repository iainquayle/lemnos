import unittest

from src.schema import Sum, Concat
from src.shared import LockedShape, OpenShape 

class TestMergeMethod(unittest.TestCase):
	def setUp(self) -> None:
		self.a = LockedShape(2, 3)
		self.b = OpenShape(3)
		self.c = OpenShape(2, 3)
		self.d = LockedShape(1, 2, 3)
		self.e = LockedShape(4, 3)
		self.f = LockedShape(6)
		self.g = LockedShape(2, 2, 3)
	def test_concat_get_output(self) -> None:
		self.assertEqual(Concat().get_output_shape([self.d, self.a]), self.g)
	def test_add_get_output(self) -> None:
		self.assertEqual(Sum().get_output_shape([self.d, self.a]), self.d)
	def test_concat_get_conformance(self) -> None:
		self.assertEqual(Concat().get_conformance_shape([self.d, self.a]), self.c)
		self.assertEqual(Concat().get_conformance_shape([]), OpenShape())
	def test_add_get_conformance(self) -> None:
		self.assertEqual(Sum().get_conformance_shape([self.d, self.a]), self.d)
		self.assertEqual(Sum().get_conformance_shape([]), OpenShape())
