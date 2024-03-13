import unittest

from src.schema import SchemaNode, Concat, Sum, JoinType, Conv, ReLU, BatchNormalization
from src.shared import Index, LockedShape, ShapeBound
from src.model import ModelNode
from src.model.model import _BuildTracker  

class TestModelNodeBuild(unittest.TestCase):
	def test_double(self):
		start_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "start")
		end_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "end")
		start_schema.add_group((end_schema, 0, JoinType.NEW))
		start = ModelNode(start_schema, LockedShape(1))
		nodes = start.attempt_build(_BuildTracker(10, {}, {start.get_schema_node(): 1}, 0), Index(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 2)
		else:
			self.fail()
	def test_split(self):
		start_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "start")
		mid_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "mid")
		end_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "end")
		start_schema.add_group((mid_schema, 0, JoinType.NEW), (end_schema, 1, JoinType.NEW))
		mid_schema.add_group((end_schema, 0, JoinType.EXISTING))
		start = ModelNode(start_schema, LockedShape(1))
		nodes = start.attempt_build(_BuildTracker(10, {}, {start.get_schema_node(): 1}, 0), Index(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 3)
			self.assertEqual({node.get_schema_node() for node in nodes}, {start_schema, mid_schema, end_schema})
		else:
			self.fail()
	def test_loop(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( (.1, 2), kernel=2, stride=2), None, None, "main")
		end = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None, None, None, "end")
		main.add_group((end, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		start = ModelNode(main, LockedShape(1, 8))
		nodes = start.attempt_build(_BuildTracker(10, {}, {start.get_schema_node(): 1}, 0), Index(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 4)
		else:
			self.fail()
	def test_split_loop(self):
		main = SchemaNode( ShapeBound((1, 1), (1, 8)), Sum(), None, None, None, "main")
		split_1 = SchemaNode( ShapeBound((1, 1), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_1")
		split_2 = SchemaNode( ShapeBound((1, 1), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), Concat(), None, None, None, "end")
		main.add_group( (split_1, 0, JoinType.NEW), (split_2, 1, JoinType.NEW))
		split_1.add_group( (main, 2, JoinType.NEW))
		split_2.add_group( (main, 2, JoinType.EXISTING))
		main.add_group( (end_node, 0, JoinType.NEW))
		start = ModelNode(main, LockedShape(1, 8))
		nodes = start.attempt_build(_BuildTracker(12, {}, {start.get_schema_node(): 1}, 0), Index(), 0)
		if nodes is not None:
			self.assertEqual(len(nodes), 11)
			id_set = set()
			for node in nodes:
				self.assertNotIn(node.get_id(), id_set)
				id_set.add(node.get_id())
		else:
			self.fail()
			

class TestModelNodeGraphFunctions(unittest.TestCase):
	def setUp(self) -> None:
		self.m1 = ModelNode(SchemaNode(ShapeBound(None, None), Sum()),  LockedShape(1, 1), LockedShape(1, 1), 0, Index())
		self.m2 = ModelNode(SchemaNode(ShapeBound(None, None), Concat()),  LockedShape(1, 2), LockedShape(1, 2), 0, Index())
		self.m3 = ModelNode(SchemaNode(ShapeBound(None, None), Concat()),  LockedShape(2, 1), LockedShape(2, 1), 0, Index())
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
		self.m1._unbind_children()
		self.assertEqual(self.m1._children, [])
		self.assertEqual(self.m2._parents, [])
	def test_unbind_parents(self) -> None:
		self.m1._add_parent(self.m2)
		self.m1._unbind_parents()
		self.assertEqual(self.m1._parents, [])
		self.assertEqual(self.m2._children, [])
	def test_set_parents(self) -> None:
		self.m1._set_parents([self.m2])
		self.assertEqual(self.m1._parents, [self.m2])
		self.assertEqual(self.m2._children, [self.m1])
		self.m1._set_parents([self.m3])
		self.assertEqual(self.m1._parents, [self.m3])
		self.assertEqual(self.m3._children, [self.m1])
	def test_has_parent_type(self) -> None:
		self.m1._add_parent(self.m2)
		self.assertTrue(self.m1.has_parent_type(self.m2.get_schema_node()))
		self.assertFalse(self.m1.has_parent_type(self.m3.get_schema_node()))
