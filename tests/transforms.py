import unittest

from torch import zeros
from torch.nn import Conv1d

from src.schema.transform import ConvParameters 
from src.shared.shape import LockedShape, OpenShape, Bound, Range
from src.shared.index import Index

#TODO: splits these tests up
class TestConvParameters(unittest.TestCase):
	def setUp(self) -> None:
		pass
	def test_input_to_output_dim(self) -> None:
		input = zeros(1, 1, 9)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Range(0.2, 2.0), kernel=3, stride=1, padding=0, dilation=1)
		self.assertFalse(parameters.validate_dimensionality(1))
		self.assertTrue(parameters.validate_dimensionality(2))
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 8)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Range(0.2, 2.0), kernel=2, stride=2, padding=0, dilation=1)
		parameters.validate_dimensionality(2)
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 9)
		shape = LockedShape(input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2, bias=False)
		parameters = ConvParameters(Range(0.2, 2.0), kernel=2, stride=1, padding=1, dilation=2)
		parameters.validate_dimensionality(2)
		self.assertEqual(parameters.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
	def test_output_to_input_dim(self) -> None:
		pass
	def test_mould_output_shape_valid(self) -> None:
		parameters = ConvParameters(Range(0.2, 2.0), kernel=2, stride=2, padding=0, dilation=1)
		parameters.validate_dimensionality(2)
		shape = LockedShape.new(2, 6)
		if (shape := parameters.get_output_shape(shape, OpenShape.new(2, 3), Bound((2, 2), (1, 8)), Index())) is not None:
			self.assertEqual(shape, LockedShape.new(2, 3))
		parameters = ConvParameters(Range(1, 2), kernel=2, stride=2, padding=0, dilation=1)
		parameters.validate_dimensionality(2)
		shape = LockedShape.new(2, 6)
		if (shape := parameters.get_output_shape(shape, OpenShape.new(3), Bound((2, 2), (1, 8)), Index())) is not None:
			self.assertEqual(shape, LockedShape.new(2, 3))
		shape = LockedShape.new(2, 6)
		if (shape := parameters.get_output_shape(shape, LockedShape.new(2, 3), Bound((1, 1000), (1, 8)), Index())) is not None:
			self.assertEqual(shape, LockedShape.new(2, 3))
	def test_output_shape_invalid(self) -> None:
		pass
