import unittest

from lemnos.schema.schema_graph import SchemaNode, _CompilationNodeStack, _CompilationTracker, _CompilationNode
from lemnos.schema.components import Concat 
from lemnos.shared import *

class TestStack(unittest.TestCase):
	def setUp(self):
		self.s1 = SchemaNode(ShapeBound(1), None, Concat())
		self.s2 = SchemaNode(ShapeBound(1), None, Concat())
		self.s3 = SchemaNode(ShapeBound(1), None, Concat())
		self.s4 = SchemaNode(ShapeBound(1), None, Concat())
	def test_record(self):
		return
		stack1 = _CompilationNodeStack(self.s1, [])
		self.assertIsNone(stack1.join_existing(self.s2, LockedShape(1), ID(0), 0))
		#stack1.join_new(self.s2, LockedShape(1), ID(0), 0)
		stack1.push(_CompilationNode({self.s2}, [ID(0)], LockedShape(1), 0))
		self.assertEqual(len(stack1), 1)
		self.assertEqual(len(stack1._stack[0].parent_ids), 1)
		self.assertIsNone(stack1.join_existing(self.s2, LockedShape(1), ID(0), 0))
		stack1.join_existing(self.s3, LockedShape(1), ID(1), 0)
		self.assertEqual(len(stack1._stack[0].parent_ids), 2)

class TestTracker(unittest.TestCase):
	def setUp(self):
		pass
		
