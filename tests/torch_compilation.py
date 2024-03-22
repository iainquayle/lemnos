import unittest

from src.schema import SchemaNode, Schema, BreedIndices, JoinType 
from src.schema.components import Concat, Sum, Conv, ReLU, BatchNormalization, Full
from src.shared import LockedShape, ShapeBound 
from src.target.torch import generate_torch_module

class TestTorchCompilation(unittest.TestCase):
	def test_split_loop(self):
		main = SchemaNode( ShapeBound(None, None), Sum(), None, None, None, "main")
		split_1 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_1")
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			Concat(), 
			Conv((.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization(), "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), Concat(), Full((0.1, 2.0)), None, None, "end")
		main.add_group( (split_1, 0, JoinType.NEW), (split_2, 1, JoinType.NEW))
		split_1.add_group( (main, 2, JoinType.NEW))
		split_2.add_group( (main, 2, JoinType.EXISTING))
		main.add_group( (end_node, 0, JoinType.NEW))
		schema = Schema([main], [end_node])
		ir = schema.compile_ir([LockedShape(1, 8)], BreedIndices(), 100)
		if ir is None:
			self.fail()
		self.assertEqual(len(ir), 11)
		print(generate_torch_module("test", ir))
