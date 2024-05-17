import unittest

from src.schema import SchemaNode, Schema, BreedIndices, New, Existing 
from src.schema.components import Sum, Conv, ReLU, BatchNorm, Full
from src.shared import LockedShape, ShapeBound, ID
from src.adapter.torch import generate_torch_module 

class TestTorchCompilation(unittest.TestCase):
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
		main.add_group( New(split_1, 0), New(split_2, 1))
		split_1.add_group( New(main, 2))
		split_2.add_group( Existing(main, 2))
		main.add_group( New(end_node, 0))
		schema = Schema([main], [end_node])
		ir = schema.compile_ir([LockedShape(1, 8)], BreedIndices(), ID(15))
		if ir is None:
			self.fail()
		print(generate_torch_module("test", ir))
		self.assertEqual(len(ir), 11)
class TestTorchModule(unittest.TestCase):
	def test_full(self):
		return
		start = SchemaNode( ShapeBound((1, 10)), None, None, None, None, None, 1, "in")
		end = SchemaNode( ShapeBound((1, 1)), None, None, Full(), None, None, 1, "out")
		start.add_group( (end, 0, JoinType.NEW))
		ir = Schema([start], [end]).compile_ir([LockedShape(10)], BreedIndices(), ID(100))
		if ir is None:
			self.fail()
		module = torch_adapter.get_module("test", ir)
		self.assertEqual(module(torch.ones(10)).shape, torch.Size([1, 1]))
	def test_conv_full(self):
		return
		start = SchemaNode( ShapeBound((1, 10), (1, 8)), None, None, Conv(kernel=2, stride=2), ReLU(), BatchNorm(), 1, "in")
		end = SchemaNode( ShapeBound((1, 1)), None, None, Full(), None, None, 1, "out")
		start.add_group( (end, 0, JoinType.NEW))
		ir = Schema([start], [end]).compile_ir([LockedShape(1, 8)], BreedIndices(), ID(100))
		if ir is None:
			self.fail()
		module = torch_adapter.get_module("test", ir)
		self.assertEqual(module(torch.ones(1, 8)).shape, torch.Size([1, 1]))
