import unittest

from src.schema.schema_graph import _CompilationNode, _CompilationNodeStack, _CompilationTracker, SchemaNode, JoinType, BreedIndices
from src.schema import SchemaNode, Concat, Sum, JoinType, Conv, ReLU, BatchNormalization 
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
		print(nodes)
	def test_loop(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( (.1, 2), kernel=2, stride=2), None, None, "main")
		end = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None, None, None, "end")
		main.add_group((end, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
		tracker = _CompilationTracker([_CompilationNodeStack(main, [_CompilationNode([], [], LockedShape(1, 8), 0)])], None, 0) 
		nodes = tracker.compile_IR(BreedIndices(), 0)
		if nodes is None:
			self.fail()
