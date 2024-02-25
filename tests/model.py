import unittest

from src.schema.schema_node import SchemaNode
from src.model.model_builder import ModelBuilder, BuildIndices
from src.shared import Bound, Range, LockedShape
from src.schema import Sum, Concat, ConvParameters, ReLU, BatchNormalization

from torch import zeros

class TestModel(unittest.TestCase):
	def test_generate_simple_module(self):
		main = SchemaNode(Bound((1, 10)), Concat())
		input_shape = LockedShape(5)
		builder = ModelBuilder([main], [main])
		model = builder.build([input_shape], BuildIndices())
		if model is not None:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			input = zeros(1, 5)
			self.assertEqual(module(input).shape, (1, 5))
		else:
			self.fail()
	def test_generate_loop_module(self):
		main = SchemaNode(Bound((1, 1), (1, 16)), Concat(), ConvParameters( Range(.1, 2), kernel=2, stride=2))
		output = SchemaNode(Bound((1, 1), (1, 1)), Concat(), None)
		main.add_group(Bound((2, 10)), (output, 0, False))
		main.add_group(Bound((2, 10)), (main, 0, False))
		input_shape = LockedShape(1, 8)
		builder = ModelBuilder([main], [output])
		model = builder.build([input_shape], BuildIndices())
		if model is not None:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			input = zeros(1, 1, 8)
			self.assertEqual(module(input).shape, (1, 1, 1))
		else: 
			self.fail()
		pass
	def test_generate_split_loop_module(self):
		main = SchemaNode( Bound((1, 1), (1, 8)), Sum())
		split_1 = SchemaNode( Bound((1, 1), (1, 8)), 
			Concat(), 
			ConvParameters(Range(.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		split_2 = SchemaNode( Bound((1, 10), (1, 8)), 
			Concat(), 
			ConvParameters(Range(.1, 2.0), kernel=2, stride=2),
			ReLU(), 
			BatchNormalization())
		end_node = SchemaNode( Bound((1, 1), (1, 1)), Concat())
		main.add_group(Bound(), (split_1, 0, False), (split_2, 1, False))
		split_1.add_group(Bound(), (main, 2, False))
		split_2.add_group(Bound(), (main, 2, True))
		main.add_group(Bound(), (end_node, 0, False))
		#when one of the split transitions is set to join on, it crashes, should atleast give reason
		builder = ModelBuilder([main], [end_node])
		model = builder.build([LockedShape(1, 8)], BuildIndices())
		if model is None:
			self.fail("Model is None")
		else:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			module.eval()
			input = zeros(1, 1, 8)
			output = module(input)
			self.assertEqual(output.shape, (1, 1, 1))
