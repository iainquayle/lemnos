import unittest

from src.schema import SchemaNode, JoinType 
from src.schema.compile import NodeTracker, NodeTrackerStack, CompilationTracker
from src.schema.components import Concat, Sum, Conv, ReLU, BatchNorm, Full
from src.schema.compile_indices import BreedIndices
from src.shared import *

class TestCompilation(unittest.TestCase):
	def test_split(self):
		start_schema = SchemaNode(ShapeBound((1, 10)), None, None, None, None, None, 1, "start")
		mid_schema = SchemaNode(ShapeBound((1, 10)), None, None, None, None, None, 1, "mid")
		end_schema = SchemaNode(ShapeBound((1, 10)), None, Concat(), None, None, None, 1, "end")
		start_schema.add_group((mid_schema, 0, JoinType.NEW), (end_schema, 1, JoinType.NEW))
		mid_schema.add_group((end_schema, 0, JoinType.EXISTING))
		tracker = CompilationTracker([NodeTrackerStack(start_schema, [NodeTracker(set(), [], LockedShape(1), 0)])], None, ID(0), ID(100)) 
		nodes = tracker.compile_ir(BreedIndices())
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 3)
	def test_divisor_hint(self):
		start_schema = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(), None, None, 1, "start")
		hinted = SchemaNode(ShapeBound((1, 10), 1), None, None, None, None, None, 2, "mid")
		end = SchemaNode(ShapeBound((1, 10), 1), None, None, Conv(groups=2), None, None, 1, "end")
		start_schema.add_group((hinted, 0, JoinType.NEW))
		hinted.add_group((end, 0, JoinType.NEW))
		tracker = CompilationTracker([NodeTrackerStack(start_schema, [NodeTracker(set(), [], LockedShape(1, 1), 0)])], None, ID(0), ID(100)) 
		nodes = tracker.compile_ir(BreedIndices())
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 3)
	def test_loop(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), None, None, Conv(kernel=2, stride=2), None, None, 1, "main")
		end = SchemaNode(ShapeBound((1, 1), (1, 1)), None, None, None, None, None, 1, "end")
		main.add_group((end, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		tracker = CompilationTracker([NodeTrackerStack(main, [NodeTracker(set(), [], LockedShape(1, 8), 0)])], None, ID(0), ID(100)) 
		nodes = tracker.compile_ir(BreedIndices())
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
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), 1, "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), None, None, Full(), None, None, 1, "end")
		main.add_group( (split_1, 0, JoinType.NEW), (split_2, 1, JoinType.NEW))
		split_1.add_group( (main, 2, JoinType.NEW))
		split_2.add_group( (main, 2, JoinType.EXISTING))
		main.add_group( (end_node, 0, JoinType.NEW))
		tracker = CompilationTracker([NodeTrackerStack(main, [NodeTracker(set(), [], LockedShape(1, 8), 0)])], None, ID(0), ID(100)) 
		nodes = tracker.compile_ir(BreedIndices())
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 11)
