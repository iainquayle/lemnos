import unittest

from lemnos.schema import SchemaNode, Schema, BreedIndices, New, Existing 
from lemnos.schema.components import Sum, Conv, ReLU, BatchNorm, Full, FlexibleConv
from lemnos.shared import LockedShape, ShapeBound, ID
from lemnos.adapter.torch import create_module, generate_source 
import torch


class TestTorchCompilation(unittest.TestCase):
	def test_split_loop(self):
		main = SchemaNode( ShapeBound(None, None), None, Sum(), None, None, None, "main")
		split_1 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), "split_1")
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), None, None, Full(), None, None, "end")
		main.add_group( New(split_1, 0), New(split_2, 1))
		split_1.add_group( New(main, 2))
		split_2.add_group( Existing(main, 2))
		main.add_group( New(end_node, 0))
		schema = Schema([main], [end_node])
		ir = schema.compile_ir([LockedShape(1, 8)], BreedIndices(), ID(15))
		if ir is None:
			self.fail()
		_ = generate_source("test", ir)
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
		main = SchemaNode( ShapeBound(None, None), None, Sum(), None, None, None, "main")
		split_1 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), "split_1")
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			None,
			None, 
			Conv(kernel=2, stride=2),
			ReLU(), 
			BatchNorm(), "split_2")
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), None, None, Full(), None, None, "end")
		main.add_group( New(split_1, 0), New(split_2, 1))
		split_1.add_group( New(main, 2))
		split_2.add_group( Existing(main, 2))
		main.add_group( New(end_node, 0))
		schema = Schema([main], [end_node])
		ir = schema.compile_ir([LockedShape(1, 8)], BreedIndices(), ID(15))
		if ir is None:
			self.fail()
		print(generate_source("Test", ir))
		module = create_module("test", ir)
		input = torch.ones(2, 1, 8)
		self.assertEqual(module(input).shape, torch.Size([2, 1]))
	def test_flex_conv(self):
		#start = SchemaNode( ShapeBound(None, None), None, None, None, None, None, "main")
		#end = SchemaNode( ShapeBound(None, None), None, Sum(), None, None, None, "end")
		conv = SchemaNode( ShapeBound((1, 10), (1, 8)), None, None, FlexibleConv(kernel=2, stride=2, groups=3), ReLU(), BatchNorm(), "conv")
		schema = Schema([conv], [conv])
		ir = schema.compile_ir([LockedShape(8, 8)], BreedIndices(), ID(1))
		if ir is None:
			self.fail()
		print(generate_source("Test", ir))
		module = create_module("test", ir)
		input = torch.ones(2, 8, 8)
		self.assertEqual(module(input).shape, torch.Size([2, 32]))
