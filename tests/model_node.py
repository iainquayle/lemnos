import unittest

from src.schema import SchemaNode, Concat 
from src.shared import Index, LockedShape, ShapeBound
from src.model import ModelNode

class TestModelNode(unittest.TestCase):
	def setUp(self) -> None:
		self.m1 = ModelNode(Index(0), 0, SchemaNode(ShapeBound((1, 1)), Concat()))
		self.m2 = ModelNode(Index(0), 1, SchemaNode(ShapeBound((1, 1)), Concat()))
		self.m3 = ModelNode(Index(0), 2, SchemaNode(ShapeBound((1, 1)), Concat()))
	def test_attempt_set_children(self) -> None:
		pass
	def test_set_children(self) -> None:
		self.m1._set_children([self.m2])
		self.assertEqual(self.m1._children, [self.m2])
		self.m1._set_children([self.m3])
		self.assertEqual(self.m1._children, [self.m3])
	def test_unbind_children(self) -> None:
		self.m1.add_child(self.m2)
		self.m1.unbind_children()
		self.assertEqual(self.m1._children, [])
		self.assertEqual(self.m2._parents, [])
	def test_unbind_parents(self) -> None:
		self.m1.add_parent(self.m2)
		self.m1.unbind_parents()
		self.assertEqual(self.m1._parents, [])
		self.assertEqual(self.m2._children, [])
	def test_set_parents(self) -> None:
		self.m1.set_parents([self.m2])
		self.assertEqual(self.m1._parents, [self.m2])
		self.assertEqual(self.m2._children, [self.m1])
		self.m1.set_parents([self.m3])
		self.assertEqual(self.m1._parents, [self.m3])
		self.assertEqual(self.m3._children, [self.m1])
