import unittest

from src.schema import SchemaNode, Concat, Sum 
from src.shared import Index, LockedShape, ShapeBound
from src.model import ModelNode

class TestModelNodeGraphFunctions(unittest.TestCase):
	def setUp(self) -> None:
		self.m1 = ModelNode(Index(0), 0, SchemaNode(ShapeBound(None, None), Sum()), LockedShape(1, 1), LockedShape(1, 1))
		self.m2 = ModelNode(Index(0), 1, SchemaNode(ShapeBound(None, None), Concat()), LockedShape(1, 2), LockedShape(1, 2))
		self.m3 = ModelNode(Index(0), 2, SchemaNode(ShapeBound(None, None), Concat()), LockedShape(2, 1), LockedShape(2, 1))
	def test_attempt_join_children_valid(self) -> None:
		self.assertTrue(self.m1.attempt_join_children([self.m2], Index(0)))
		self.assertTrue(self.m3.attempt_join_children([self.m2], Index(0)))
	def test_attempt_join_children_invalid(self) -> None:
		self.assertTrue(self.m2.attempt_join_children([self.m1], Index(0)))
		self.assertFalse(self.m3.attempt_join_children([self.m1], Index(0)))
	def test_set_children(self) -> None:
		self.m1._set_children([self.m2])
		self.assertEqual(self.m1._children, [self.m2])
		self.m1._set_children([self.m3])
		self.assertEqual(self.m1._children, [self.m3])
	def test_unbind_children(self) -> None:
		self.m1._add_child(self.m2)
		self.m1.unbind_children()
		self.assertEqual(self.m1._children, [])
		self.assertEqual(self.m2._parents, [])
	def test_unbind_parents(self) -> None:
		self.m1._add_parent(self.m2)
		self.m1.unbind_parents()
		self.assertEqual(self.m1._parents, [])
		self.assertEqual(self.m2._children, [])
	def test_set_parents(self) -> None:
		self.m1._set_parents([self.m2])
		self.assertEqual(self.m1._parents, [self.m2])
		self.assertEqual(self.m2._children, [self.m1])
		self.m1._set_parents([self.m3])
		self.assertEqual(self.m1._parents, [self.m3])
		self.assertEqual(self.m3._children, [self.m1])
