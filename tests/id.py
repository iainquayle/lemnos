import unittest

from src.shared import ID

class TestID(unittest.TestCase):
	def test_new(self):
		self.assertRaises(ValueError, ID, -1)
		self.assertEqual(ID(0), 0)
		self.assertEqual(ID(1), 1)
	def test_add(self):
		self.assertRaises(ValueError, ID(0).__add__, -1)
		self.assertEqual(ID(0) + 1, 1)
		self.assertEqual(ID(1) + 1, 2)
	def test_sub(self):
		self.assertRaises(ValueError, ID(0).__sub__, 1)
		self.assertEqual(ID(1) - 1, 0)
		self.assertEqual(ID(2) - 1, 1)
	def test_mul(self):
		self.assertRaises(ValueError, ID(1).__mul__, -1)
		self.assertEqual(ID(0) * 1, 0)
		self.assertEqual(ID(1) * 2, 2)
	def test_floordiv(self):
		self.assertRaises(ValueError, ID(1).__floordiv__, -1)
		self.assertEqual(ID(1) // 1, 1)
		self.assertEqual(ID(2) // 2, 1)
