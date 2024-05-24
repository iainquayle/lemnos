import unittest

from src.schema import *
from src.schema.components import *
from src.shared import *

class TestSchemaGraph(unittest.TestCase):
	def setUp(self):
		self.sn_1 = SchemaNode(ShapeBound((1, 10), 1), None, Conv(groups=2), None, None, 1)
		self.sn_2 = SchemaNode(ShapeBound((1, 10), 1), None, Conv(groups=2), None, None, 3)
		self.sn_3 = SchemaNode(ShapeBound((1, 10), 1), None, Conv(groups=3), GLU(), None, 5)
	def test_get_divisor(self):
		self.assertEqual(self.sn_1.get_divisor(), 2)
		self.assertEqual(self.sn_2.get_divisor(), 6)
		self.assertEqual(self.sn_3.get_divisor(), 30)
	def test_get_output_shape_valid(self):
		self.assertEqual(self.sn_1.get_output_shape(LockedShape(2, 1), OpenShape(), 1, CompileIndex()), LockedShape(2, 1))
		self.assertEqual(self.sn_2.get_output_shape(LockedShape(2, 1), OpenShape(), 1, CompileIndex()), LockedShape(6, 1))
