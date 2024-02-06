import unittest

from src.shared.shape import Shape, LockedShape, OpenShape, Bound, Range

class TestShapeCommons(unittest.TestCase):
	def setUp(self) -> None:
		self.a = LockedShape.new(2, 3)
		self.b = OpenShape.new(3)
		self.c = OpenShape.new(2, 3)
		self.d = LockedShape.new(1, 2, 3)
		self.e = LockedShape.new(4, 3)
		self.f = LockedShape.new(3, 3)
	def test_length(self) -> None:
		self.assertEqual(len(self.a), 2)
	def test_upper_length(self) -> None:
		self.assertEqual(self.a.upper_length(), 1)
		self.assertEqual(self.c.upper_length(), 2)
	def test_squash(self) -> None:
		self.assertEqual(self.d.squash(2), self.a)
		self.assertEqual(self.d.squash(3), self.d)
		self.assertEqual(self.c.squash(2), self.b)
		self.assertEqual(self.c.squash(3), self.c)
	def test_dimensionality(self) -> None:
		self.assertEqual(self.a.dimensionality(), 2)
		self.assertEqual(self.c.dimensionality(), 3)
	def test_both_locked_lossless(self) -> None:
		self.assertEqual(self.a.common_lossless(self.a), self.a)
		self.assertEqual(self.a.common_lossless(self.d), self.d)
		self.assertIsNone(self.a.common_lossless(self.e))
		self.assertIsNone(self.e.common_lossless(self.d))
	def test_one_locked_lossless(self) -> None:
		self.assertEqual(self.a.common_lossless(self.b), self.a)
		self.assertEqual(self.a.common_lossless(self.c), self.d)
		self.assertIsNone(self.a.common_lossless(self.f))
	def test_both_open_lossless(self) -> None:
		self.assertEqual(self.b.common_lossless(self.b), self.b)
		self.assertEqual(self.b.common_lossless(self.c), self.c)
	def test_reduce_common_lossless(self) -> None:
		self.assertEqual(Shape.reduce_common_lossless([self.a, self.a, self.b]), self.a)
		self.assertEqual(Shape.reduce_common_lossless([self.a, self.d, self.c]), self.d)
		self.assertIsNone(Shape.reduce_common_lossless([self.a, self.a, self.e]))
	def test_both_constrained_keep_dimensionality(self) -> None:
		self.assertEqual(self.a.common(self.a), self.a)
		self.assertEqual(self.a.common(self.d), self.a)
		self.assertEqual(self.d.common(self.a), self.d)
		self.assertIsNone(self.a.common(self.e))
		self.assertIsNone(self.e.common(self.d))	

class TestBounds(unittest.TestCase):
	def setUp(self) -> None:
		self.bound = Bound((1, 10))
	def test_contains_in(self) -> None:
		self.assertTrue(LockedShape.new(5) in self.bound)
		self.assertTrue(LockedShape.new(1, 5) in self.bound)
	def test_contains_out(self) -> None:
		self.assertFalse(LockedShape.new(11) in self.bound)
	
