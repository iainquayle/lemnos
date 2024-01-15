import unittest

from src.shared.shape import Shape, Bound, Range

class TestShapeCommons(unittest.TestCase):
	def setUp(self) -> None:
		self.a = Shape.new_fixed([2, 3])
		self.b = Shape.new_unfixed([3])
		self.c = Shape.new_unfixed([2, 3])
		self.d = Shape.new_fixed([1, 2, 3])
		self.e = Shape.new_fixed([4, 3])
		self.f = Shape.new_fixed([3, 3])
	def test_both_constrained(self) -> None:
		self.assertEqual(Shape.reduce_collection([self.a, self.a]), self.a)
		self.assertEqual(Shape.reduce_collection([self.a, self.d]), self.d)
		self.assertIsNone(Shape.reduce_collection([self.a, self.e]))
	def test_one_constrained(self) -> None:
		self.assertEqual(Shape.reduce_collection([self.a, self.b]), self.a)
		self.assertEqual(Shape.reduce_collection([self.a, self.c]), self.d)
		self.assertEqual(Shape.reduce_collection([self.a, Shape(2, Size())]), self.a)
		self.assertIsNone(Shape.reduce_collection([self.a, self.f]))
	def test_none_constrained(self) -> None:
		self.assertEqual(Shape.reduce_collection([self.b, self.b]), self.b)
		self.assertEqual(Shape.reduce_collection([self.b, self.c]), self.c)
		self.assertEqual(Shape.reduce_collection([self.b, Shape(2, Size())]), self.b)
		pass
