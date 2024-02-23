import unittest

from src.schema.schema_node import SchemaNode
from src.model.model_builder import ModelBuilder, BuildIndices
from src.shared.shape import Bound, Range, LockedShape
from src.schema.merge_method import Sum, Concat
from src.schema.transform import ConvParameters
from src.schema.activation import ReLU
from src.schema.regularization import BatchNormalization 

from torch import zeros

class TestModel(unittest.TestCase):
	def test_generated_module(self):
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
			ordered_nodes = model.get_ordered_nodes()
			node_set = set()
			for node in ordered_nodes:
				if node in node_set:
					self.fail("Duplicate node in ordered list")
				node_set.add(node)
			self.assertEqual(len(ordered_nodes), 11)
			#print(model.to_torch_module_src("Test"))
			return
			module = model.get_torch_module_handle("Test")()
			input = zeros(1, 1, 8)
			output = module(input)
			self.assertEqual(output.shape, (1, 1, 1))
