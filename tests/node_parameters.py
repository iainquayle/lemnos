import unittest

from torch import zeros
from torch.nn import Conv1d

from src.pattern.node_parameters import IdentityParameters, ConvParameters 
from src.shared.shape import LockedShape, OpenShape, Bound, Range

class TestIdentityParemeters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters(Bound([(1, 8), (1, 8)]))
	def test_mould_output_shape_valid(self) -> None:
		shapes = self.parameters.get_mould_and_output_shapes(LockedShape.new(1, 1, 8), OpenShape.new(1, 1, 8))
		if shapes is not None:
			mould, output = shapes
			self.assertEqual(mould, LockedShape.new(1, 8))
			self.assertEqual(output, LockedShape.new(1, 8))
	def test_output_shape_invalid(self) -> None:
		shapes = self.parameters.get_mould_and_output_shapes(LockedShape.new(1, 1, 8), OpenShape.new(1, 4, 8))
		self.assertIsNone(shapes)

class TestConvParameters(unittest.TestCase):
	def setUp(self) -> None:
		pass
	def test_input_to_output_dim(self) -> None:
		input = zeros(1, 1, 9)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Bound([(1, 1), (1, 9)]), Range(0.2, 2.0), kernel=3, stride=1, padding=0, dilation=1)
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 8)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Bound([(1, 1), (1, 9)]), Range(0.2, 2.0), kernel=2, stride=2, padding=0, dilation=1)
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 9)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2, bias=False)
		parameters = ConvParameters(Bound([(1, 1), (1, 9)]), Range(0.2, 2.0), kernel=2, stride=1, padding=1, dilation=2)
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
	def test_output_to_input_dim(self) -> None:
		pass
	def test_mould_output_shape_valid(self) -> None:
		parameters = ConvParameters(Bound([(1, 1), (1, 8)]), Range(0.2, 2.0), kernel=2, stride=2, padding=0, dilation=1)
		shape = LockedShape.new(2, 6)
		shapes = parameters.get_mould_and_output_shapes(shape, OpenShape.new(2, 3))
		if shapes is not None:
			_, output = shapes
			self.assertEqual(output, LockedShape.new(1, 3))
		parameters = ConvParameters(Bound([(1, 1000), (1, 8)]), Range(1, 2), kernel=2, stride=2, padding=0, dilation=1)
		shapes = parameters.get_mould_and_output_shapes(shape, OpenShape.new(3))
		if shapes is not None:
			_, output = shapes
			self.assertGreater(output[0], 1)
		shapes = parameters.get_mould_and_output_shapes(shape, LockedShape.new(2, 3))
		if shapes is not None:
			_, output = shapes
			self.assertEqual(output, LockedShape.new(2, 3))
	def test_output_shape_invalid(self) -> None:
		parameters = ConvParameters(Bound([(1, 1), (1, 9)]), Range(0.2, 2.0), kernel=2, stride=2, padding=0, dilation=1)
		shape = LockedShape.new(2, 6)
		self.assertIsNone(parameters.get_mould_and_output_shapes(shape, LockedShape.new(2, 3)))
		self.assertIsNone(parameters.get_mould_and_output_shapes(shape, OpenShape.new(1, 6)))
		shape = LockedShape.new(2, 7)
		self.assertIsNone(parameters.get_mould_and_output_shapes(shape, LockedShape.new(2, 3)))
		
