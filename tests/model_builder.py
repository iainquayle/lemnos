import unittest   

from src.model.model_builder import _ExpansionNode, _ExpansionStack, _ExpansionCollection
from src.model.model import ModelNode
from src.schema.schema_node import SchemaNode, Transition
from src.schema.node_parameters import IdentityParameters 
from src.shared.merge_method import Concat
from src.shared.shape import Bound, LockedShape
from src.shared.index import Index
from copy import copy

s1 = SchemaNode(IdentityParameters(Bound()), Concat())
s2 = SchemaNode(IdentityParameters(Bound()), Concat())
shape = LockedShape.new(1)
m1s1 = ModelNode(Index(), 0, s1, shape, shape, [])
m2s1 = ModelNode(Index(), 0, s1, shape, shape, [])
m1s2 = ModelNode(Index(), 0, s2, shape, shape, [])
m2s2 = ModelNode(Index(), 0, s2, shape, shape, [])

class TestExpansionCollection(unittest.TestCase):
	def setUp(self) -> None:
		self.node1 = _ExpansionNode([m1s1], 5)
		self.node2 = _ExpansionNode([m1s2], 10)
		self.stack1 = _ExpansionStack([self.node1])
		self.stack2 = _ExpansionStack([self.node2])
		self.collection = _ExpansionCollection({s1: self.stack1, s2: self.stack2})
	def test_build_min_final(self):
		pass
	def test_build_min_fail_join_existing(self):
		pass
	def test_build_min_fail_parameters(self):
		pass
	def test_build_min_fail_child(self):
		pass
	def test_pop_min_full(self):
		self.assertEqual(self.collection.pop_min(), (s1, self.node1))
		self.assertEqual(self.collection.pop_min(), (s2, self.node2))
	def test_pop_min_empty(self):
		collection = _ExpansionCollection()
		self.assertIsNone(collection.pop_min())
	def test_copy(self):
		new_collection = copy(self.collection) 
		self.assertEqual(len(new_collection), len(self.collection))
		self.assertEqual(new_collection[s1].peek().get_parents(), self.collection[s1].peek().get_parents())
		self.assertEqual(new_collection[s2].peek().get_parents(), self.collection[s2].peek().get_parents())
		self.assertNotEqual(id(new_collection[s1]), id(self.collection[s1]))
		self.assertNotEqual(id(new_collection[s2]), id(self.collection[s2]))
		self.assertNotEqual(id(new_collection[s1].peek()), id(self.collection[s1].peek()))
		self.assertNotEqual(id(new_collection[s2].peek()), id(self.collection[s2].peek()))
	def test_record_valid(self):
		t2_nj = Transition(s2, 1)
		self.assertTrue(self.collection.record_transition(t2_nj, m1s1))
		self.assertEqual(self.collection[s2].peek().get_parents(), [m1s1])
		self.assertEqual(self.collection[s2].peek().get_priority(), 1)
		t2_j = Transition(s2, 0, True)
		self.assertTrue(self.collection.record_transition(t2_j, m2s2))
		self.assertTrue(m2s2 in self.collection[s2].peek().get_parents())
		self.assertEqual(self.collection[s2].peek().get_priority(), 0)
	def test_record_invalid(self):
		t2_j = Transition(s2, 1, True)
		collection = _ExpansionCollection({s1: self.stack1, s2: _ExpansionStack([])})
		self.assertFalse(collection.record_transition(t2_j, m1s1))
		t1_j = Transition(s1, 0, True)
		self.assertFalse(collection.record_transition(t1_j, m1s1))

class TestExpansionNode(unittest.TestCase):
	def setUp(self) -> None:
		self.node = _ExpansionNode([m1s1], 0)
	def test_available(self):
		self.assertFalse(self.node.available(m2s1))
		self.assertTrue(self.node.available(m1s2))

class TestExpansionStack(unittest.TestCase):
	def setUp(self) -> None:
		self.node1 = _ExpansionNode([m1s1, m1s2], 0)
		self.node2 = _ExpansionNode([m1s2], 0)
		self.stack = _ExpansionStack([self.node1, self.node2])
	def test_available(self):
		self.assertEqual(self.stack.get_available(m2s1), self.node2)
	def test_available_none(self):
		self.assertIsNone(self.stack.get_available(m2s2))
	def test_priority(self):
		self.assertEqual(self.stack.get_priority(), 0)
		self.stack.push(_ExpansionNode([m1s2], 1))
		self.assertEqual(self.stack.get_priority(), 1)
		self.assertEqual(_ExpansionStack().get_priority(), Transition.get_max_priority() + 1)

