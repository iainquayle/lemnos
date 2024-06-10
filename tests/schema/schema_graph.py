import unittest

from lemnos.schema import *
from lemnos.schema.components import *
from lemnos.shared import *
from lemnos.schema.schema_graph import Conformance

class TestSchemaGraph(unittest.TestCase):
	def setUp(self):
		self.sn_1 = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(groups=2), None, None, 1)
		self.sn_2 = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(groups=2), None, None, 3)
		self.sn_3 = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(groups=3), GLU(), None, 5)
	def test_get_divisor(self):
		conformance = self.sn_1.get_conformance([])
		if conformance is None:
			self.fail()
		self.assertEqual(conformance.divisor, 2)
		conformance = self.sn_2.get_conformance([])
		if conformance is None:
			self.fail()
		self.assertEqual(conformance.divisor, 6)
		conformance = self.sn_3.get_conformance([])
		if conformance is None:
			self.fail()
		self.assertEqual(conformance.divisor, 30)
	def test_get_output_shape_valid(self):
		self.assertEqual(self.sn_1.get_output_shape(LockedShape(2, 1), Conformance(OpenShape(), 1), CompilationIndex()), LockedShape(2, 1))
		self.assertEqual(self.sn_2.get_output_shape(LockedShape(2, 1), Conformance(OpenShape(), 1), CompilationIndex()), LockedShape(6, 1))
