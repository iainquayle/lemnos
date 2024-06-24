import unittest

from lemnos.schema.schema_graph import _CompilationNode, _CompilationNodeStack, _CompilationTracker 
from lemnos.schema.schema_graph import *
from lemnos.schema.components import Concat, Sum, Conv, ReLU, BatchNorm, Full
from lemnos.schema.compilation_indices import BreedIndices
from lemnos.shared import *

class Test_Compilation(unittest.TestCase):
	def test_split(self):
		start_schema = SchemaNode(ShapeBound((1, 10)), None, None, None, None, None, 1, "start")
		mid_schema = SchemaNode(ShapeBound((1, 10)), None, None, None, None, None, 1, "mid")
		end_schema = SchemaNode(ShapeBound((1, 10)), None, Concat(), None, None, None, 1, "end")
		start_schema.add_group(New(mid_schema, 0), New(end_schema, 1))
		mid_schema.add_group(Existing(end_schema, 0))
		tracker = _CompilationTracker([_CompilationNodeStack(start_schema, [_CompilationNode(set(), [], LockedShape(5), 0)])], None) 
		schema, node = tracker.pop_min()
		nodes = schema._compile(node, tracker, BreedIndices(), ID(0), ID(5))
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 3)
	def test_divisor_hint(self):
		start_schema = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(), None, None, 1, "start")
		hinted = SchemaNode(ShapeBound((1, 10), 1), None, None, None, None, None, 2, "hinted")
		end = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(groups=2), None, None, 1, "end")
		start_schema.add_group(New(hinted, 0))
		hinted.add_group(New(end, 0))
		tracker = _CompilationTracker([_CompilationNodeStack(start_schema, [_CompilationNode(set(), [], LockedShape(1, 1), 0)])], None) 
		schema, node = tracker.pop_min()
		nodes = schema._compile(node, tracker, BreedIndices(), ID(0), ID(5))
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 3)
	def test_loop(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), None, None, Conv(kernel=2, stride=2), None, None, 1, "main")
		end = SchemaNode(ShapeBound((1, 1), (1, 1)), None, None, None, None, None, 1, "end")
		main.add_group(New(end, 0))
		main.add_group(New(main, 0))
		tracker = _CompilationTracker([_CompilationNodeStack(main, [_CompilationNode(set(), [], LockedShape(1, 8), 0)])], None) 
		schema, node = tracker.pop_min()
		nodes = schema._compile(node, tracker, BreedIndices(), ID(0), ID(5))
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 4)
	def test_split_loop(self):
		main = SchemaNode( ShapeBound(None, None), None, Sum(), None, None, None, 1, "main")
		split_1 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), 1, "split_1")
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2, mix_groups=True),
			ReLU(), 
			BatchNorm(), 1, "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), None, None, Full(), None, None, 1, "end")
		main.add_group( New(split_1, 0), New(split_2, 1))
		split_1.add_group( New(main, 2))
		split_2.add_group( Existing(main, 2))
		main.add_group( New(end_node, 0))
		tracker = _CompilationTracker([_CompilationNodeStack(main, [_CompilationNode(set(), [], LockedShape(1, 8), 0)])], None)   
		schema, node = tracker.pop_min()
		nodes = schema._compile(node, tracker, BreedIndices(), ID(0), ID(12))
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 11)
