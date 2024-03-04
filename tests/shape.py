import unittest

from src.shared import Shape, LockedShape, OpenShape, ShapeBound 

class TestShapeCommons(unittest.TestCase):
	def setUp(self) -> None:
		self.a = LockedShape(2, 3)
		self.b = OpenShape(3)
		self.c = OpenShape(2, 3)
		self.d = LockedShape(1, 2, 3)
		self.e = LockedShape(4, 3)
		self.f = LockedShape(3, 3)
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
	def test_reverse_index(self) -> None: #something weird happening
		self.assertTrue(isinstance(self.a[-1], int))
		self.assertEqual(self.a[-1], 3)
	def test_diff(self) -> None:
		self.assertEqual(LockedShape(1, 1).upper_difference(LockedShape(1, 2)), 2)

class TestShapeBounds(unittest.TestCase):
	def setUp(self) -> None:
		self.bound = ShapeBound((1, 10))
	def test_contains_in(self) -> None:
		self.assertTrue(LockedShape(5) in self.bound)
		self.assertTrue(LockedShape(1, 5) in self.bound)
	def test_contains_out(self) -> None:
		self.assertFalse(LockedShape(11) in self.bound)
	
