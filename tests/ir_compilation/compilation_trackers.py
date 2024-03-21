import unittest

from src.schema.schema_graph import SchemaNode, JoinType
from src.schema.ir_compilation import NodeTracker, NodeTrackerStack, CompilationTracker
from src.schema.components import Concat 
from src.shared import *

class TestStack(unittest.TestCase):
	def setUp(self):
		self.s1 = SchemaNode(ShapeBound(1), Concat())
		self.s2 = SchemaNode(ShapeBound(1), Concat())
		self.s3 = SchemaNode(ShapeBound(1), Concat())
		self.s4 = SchemaNode(ShapeBound(1), Concat())
	def test_record(self):
		stack1 = NodeTrackerStack(self.s1, [])
		self.assertRaises(ValueError, stack1.next, self.s2, JoinType.EXISTING, LockedShape(1), 0, 0)
		stack2 = stack1.next(self.s2, JoinType.NEW, LockedShape(1), 0, 0)
		self.assertEqual(len(stack1), 0)
		self.assertEqual(len(stack2), 1)
		stack3 = stack2.next(self.s2, JoinType.NEW, LockedShape(1), 1, 0)
		stack4 = stack3.next(self.s3, JoinType.EXISTING, LockedShape(1), 2, 0)
		self.assertEqual(len(stack3._stack[1].parent_ids), 1)
		self.assertEqual(len(stack4._stack[1].parent_ids), 2)
		self.assertEqual(len(stack4._stack[0].parent_ids), 1)
		self.assertEqual(stack4._stack[1].input_shape, LockedShape(2))
		stack5 = stack4.next(self.s3, JoinType.EXISTING, LockedShape(1), 3, 0)
		self.assertEqual(len(stack5._stack[0].parent_ids), 2)

class TestTracker(unittest.TestCase):
	def setUp(self):
		self.s1 = SchemaNode(ShapeBound(1), Concat())
		self.s2 = SchemaNode(ShapeBound(1), Concat())
		self.s3 = SchemaNode(ShapeBound(1), Concat())
		self.s4 = SchemaNode(ShapeBound(1), Concat())
		
