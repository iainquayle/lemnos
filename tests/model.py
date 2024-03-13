import unittest

from src.shared import ShapeBound, LockedShape
from src.schema import Schema, SchemaNode, Sum, Concat, Conv, ReLU, BatchNormalization, Full, Sigmoid, Softmax, JoinType
from src.model import Model, BreedIndices 

from torch import zeros, tensor
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam

class TestModel(unittest.TestCase):
	def test_generate_simple_module(self):
		return
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
		return
		main = SchemaNode(ShapeBound((1, 1), (1, 16)), Concat(), Conv( (.1, 2), kernel=2, stride=2))
		output = SchemaNode(ShapeBound((1, 1), (1, 1)), Concat(), None)
		main.add_group((output, 0, JoinType.NEW))
		main.add_group((main, 0, JoinType.NEW))
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
		schema = Schema([main], [end_node])
		model = Model(schema, [LockedShape(1, 8)], BreedIndices())
		if model is None:
			self.fail("Model is None")
		else:
			module_handle = model.get_torch_module_handle("Test")
			module = module_handle()
			module.eval()
			input = zeros(1, 1, 8)
			output = module(input)
			self.assertEqual(output.shape, (1, 1, 1))
	def test_module_function(self):
		return
		first = SchemaNode(ShapeBound((3, 3)), Concat(), Full((.1, 2)), Sigmoid())
		hidden = SchemaNode(ShapeBound((2, 2)), Concat(), Full((.1, 2)), Sigmoid())
		final = SchemaNode(ShapeBound((3, 3)), Concat(), Full((.1, 2)), Softmax())
		first.add_group((hidden, 0, JoinType.NEW))
		hidden.add_group((final, 0, JoinType.NEW))
		schema = Schema([first], [final])
		model = schema.build([LockedShape(3)], BreedIndices())
		if model is None:
			self.fail()
		else:
			module = model.get_torch_module_handle("Test")()
			inputs = tensor([
				[1.0, 0.0, 0.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 1.0],])
			truths = tensor([
				[1.0, 0.0, 0.0],
				[0.0, 1.0, 0.0],
				[0.0, 0.0, 1.0],])
			#module = nn.Sequential( nn.Linear(3, 3, bias=True), nn.Sigmoid(), nn.Linear(3, 3, bias=True), nn.Sigmoid(), nn.Linear(3, 3, bias=True))
			module.train()
			optimizer = Adam(module.parameters(), lr=0.01)
			criterion = CrossEntropyLoss()
			for _ in range(1000):
				optimizer.zero_grad()
				output = module(inputs)
				loss = criterion(output, truths)
				loss.backward()
				optimizer.step()
			module.eval()
			for i in range(3):
				output = module(inputs[i])
				self.assertTrue(output.argmax() == i)
