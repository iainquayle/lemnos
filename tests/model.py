import unittest

from src.shared import ShapeBound, Range, LockedShape
from src.schema import Schema, SchemaNode, BreedIndices, Sum, Concat, Conv, ReLU, BatchNormalization 

from torch import zeros

class TestModel(unittest.TestCase):
	def test_generate_simple_module(self):
		main = SchemaNode(ShapeBound((1, 10)), Concat())
		input_shape = LockedShape(5)
		builder = Schema([main], [main])
		model = builder.build([input_shape], BreedIndices())
		if model is not None:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			input = zeros(1, 5)
			self.assertEqual(module(input).shape, (1, 5))
		else:
			self.fail()
	def test_generate_loop_module(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( Range(.1, 2), kernel=2, stride=2))
		output = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None)
		main.add_group(ShapeBound((2, 10)), (output, 0, False))
		main.add_group(ShapeBound((2, 10)), (main, 0, False))
		input_shape = LockedShape(1, 8)
		builder = Schema([main], [output])
		model = builder.build([input_shape], BreedIndices())
		if model is not None:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			input = zeros(1, 1, 8)
			self.assertEqual(module(input).shape, (1, 1, 1))
		else: 
			self.fail()
		pass
	def test_generate_split_loop_module(self):
		main = SchemaNode( ShapeBound((1, 1), (1, 8)), Sum())
		split_1 = SchemaNode( ShapeBound((1, 1), (1, 8)), 
			Concat(), 
			Conv(Range(.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		split_2 = SchemaNode( ShapeBound((1, 10), (1, 8)), 
			Concat(), 
			Conv(Range(.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		end_node = SchemaNode( ShapeBound((1, 1), (1, 1)), Concat())
		main.add_group(ShapeBound(), (split_1, 0, False), (split_2, 1, False))
		split_1.add_group(ShapeBound(), (main, 2, False))
		split_2.add_group(ShapeBound(), (main, 2, True))
		main.add_group(ShapeBound(), (end_node, 0, False))
		#when one of the split transitions is set to join on, it crashes, should atleast give reason
		builder = Schema([main], [end_node])
		model = builder.build([LockedShape(1, 8)], BreedIndices())
		if model is None:
			self.fail("Model is None")
		else:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			module.eval()
			input = zeros(1, 1, 8)
			output = module(input)
			self.assertEqual(output.shape, (1, 1, 1))
