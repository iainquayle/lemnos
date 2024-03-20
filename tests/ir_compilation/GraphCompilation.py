import unittest

from src.schema.schema_graph import SchemaNode, JoinType
from src.schema.ir_compilation import _CompilationNode, _CompilationNodeStack, _CompilationTracker
from src.schema import SchemaNode, Concat, Sum, JoinType, Conv, ReLU, BatchNormalization 
from src.schema.compilation_indices import BreedIndices
from src.schema.merge_method import *
from src.shared import *

class TestCompilation(unittest.TestCase):
	def test_split(self):
		start_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "start")
		mid_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "mid")
		end_schema = SchemaNode(ShapeBound((1, 10)), Concat(), None, None, None, "end")
		start_schema.add_group((mid_schema, 0, JoinType.NEW), (end_schema, 1, JoinType.NEW))
		mid_schema.add_group((end_schema, 0, JoinType.EXISTING))
		tracker = _CompilationTracker([_CompilationNodeStack(start_schema, [_CompilationNode([], [], LockedShape(1), 0)])], None, 0) 
		nodes = tracker.compile_IR(BreedIndices(), 0)
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 3)
	def test_loop(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( (.1, 2), kernel=2, stride=2), None, None, "main")
		end = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None, None, None, "end")
		main.add_group((end, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		tracker = _CompilationTracker([_CompilationNodeStack(main, [_CompilationNode([], [], LockedShape(1, 8), 0)])], None, 0) 
		nodes = tracker.compile_IR(BreedIndices(), 0)
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 4)
	def test_split_loop(self):
		main = SchemaNode( ShapeBound((1, 1), (1, 8)), Sum(), None, None, None, "main")
		split_1 = SchemaNode( ShapeBound((1, 1), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_1")
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), Concat(), None, None, None, "end")
		main.add_group( (split_1, 0, JoinType.NEW), (split_2, 1, JoinType.NEW))
		split_1.add_group( (main, 2, JoinType.NEW))
		split_2.add_group( (main, 2, JoinType.EXISTING))
		main.add_group( (end_node, 0, JoinType.NEW))
		tracker = _CompilationTracker([_CompilationNodeStack(main, [_CompilationNode([], [], LockedShape(1, 8), 0)])], None, 0) 
		nodes = tracker.compile_IR(BreedIndices(), 0)
		if nodes is None:
			self.fail()
		self.assertEqual(len(nodes), 11)
