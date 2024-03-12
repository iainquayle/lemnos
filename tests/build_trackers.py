import unittest

from src.schema import SchemaNode, Concat, Transition, JoinType
from src.shared import Index, LockedShape, ShapeBound
from src.model import ModelNode
from src.model.model_node import _BuildTracker, _BuildStack

class TestStack(unittest.TestCase):
	def setUp(self) -> None:
		self.s1 = SchemaNode(ShapeBound(None), Concat())
		self.s2 = SchemaNode(ShapeBound(None), Concat())
		self.stack = _BuildStack(self.s1, [])
		self.n1 = ModelNode(self.s1)
		self.n2 = ModelNode(self.s2)
	def test_record_and_get_new(self) -> None:
		node = self.stack.record_and_get(self.n1, JoinType.NEW, 0)
		node2 = self.stack.record_and_get(self.n2, JoinType.NEW, 0)
		self.assertNotEqual(node, node2)
	def test_record_and_get_existing_none(self) -> None:
		self.assertIsNone(self.stack.record_and_get(self.n1, JoinType.EXISTING, 0))
		node = self.stack.record_and_get(self.n1, JoinType.NEW, 0)
		if node is not None:
			node._add_parent(self.n1)
			self.assertIsNone(self.stack.record_and_get(self.n1, JoinType.EXISTING, 0))
		else:
			self.fail("node is None")
	def test_record_and_get_existing(self) -> None:
		node = self.stack.record_and_get(self.n1, JoinType.NEW, 0)
		if node is not None:
			node._add_parent(self.n1)
			self.assertEqual(node, self.stack.record_and_get(self.n2, JoinType.EXISTING, 0))
		else:
			self.fail("node is None")
	def test_record_and_get_auto(self) -> None:
		node = self.stack.record_and_get(self.n1, JoinType.AUTO, 0)
		if node is not None:
			node._add_parent(self.n1)
			self.assertEqual(node, self.stack.record_and_get(self.n2, JoinType.AUTO, 0))
			node._add_parent(self.n2)
			self.assertNotEqual(node, self.stack.record_and_get(self.n2, JoinType.AUTO, 0))
		else:
			self.fail("node is None")

class TestBuildTracker(unittest.TestCase):
	def setUp(self):
		self.s1 = SchemaNode(ShapeBound(None), Concat())
		self.s2 = SchemaNode(ShapeBound(None), Concat())
		self.n1_1 = ModelNode(self.s1)
		self.n1_2 = ModelNode(self.s1)
		self.n2_1 = ModelNode(self.s2)
		self.n2_2 = ModelNode(self.s2)
	pass
