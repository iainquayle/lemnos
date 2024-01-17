import unittest

from torch import Size, zeros
from torch.nn import Conv1d

from src.pattern.node_parameters import IdentityParameters, ConvParameters 
from src.shared.shape import LockedShape, OpenShape, Bound, Range
from src.shared.merge_method import Concat, Add

from math import prod

class TestBaseParameters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.merge_method = Concat()
		self.base_shape = Size([2, 4])
	def test_get_mould_shape_same_dim(self) -> None:
		self.parameters.shape_bounds = Bound([1, 1], [4, 4])
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape, self.base_shape]), Size([4, 4]))
	def test_get_mould_shape_up_dim(self) -> None:
		self.parameters.shape_bounds = Bound([1, 1, 1], [4, 4, 4])
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape]), Size([1, 2, 4]))
	def test_get_mould_shape_down_dim(self) -> None:
		self.parameters.shape_bounds = Bound([1], [4])
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape]), Size([8]))

class TestIdentityParemeters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.shape_bounds = Bound([1, 1], [4, 8])
		self.parameters.merge_method = Concat()
		self.base_shape = Size([2, 4])
	def test_transform_src(self) -> None:
		self.assertEqual(self.parameters.get_transform_src(self.base_shape, self.base_shape), "Identity()")
	def test_output_shape(self) -> None:
		self.assertEqual(self.parameters.get_output_shape(self.base_shape, ConformanceShape(2, Size([2, 4]))), Size([2, 4]))
		self.assertIsNone(self.parameters.get_output_shape(self.base_shape, ConformanceShape(2, Size([2, 5]))))

class TestConvParameters(unittest.TestCase):
	def setUp(self) -> None:
		pass
	def test_validate_output_shape(self) -> None:
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=3, stride=1, padding=0, dilation=1)
		pass
	def test_input_to_output_dim(self) -> None:
		input = zeros(1, 1, 9)
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=3, stride=1, padding=0, dilation=1)
		self.assertEqual(parameters.input_dim_to_output_dim(input.shape[1:], 1), reference(input).shape[2])
		input = zeros(1, 1, 8)
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=2, stride=2, padding=0, dilation=1)
		self.assertEqual(parameters.input_dim_to_output_dim(input.shape[1:], 1), reference(input).shape[2])
		input = zeros(1, 1, 9)
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2, bias=False)
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=2, stride=1, padding=1, dilation=2)
		self.assertEqual(parameters.input_dim_to_output_dim(input.shape[1:], 1), reference(input).shape[2])
	def test_output_to_input_dim(self) -> None:
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=3, stride=1, padding=0, dilation=1)
		self.assertEqual(parameters.output_dim_to_input_dim(Size([1, 7]), 1), 9)
		parameters = ConvParameters(Bound([1, 1], [1, 9]), kernel=2, stride=2, padding=0, dilation=1)
		self.assertEqual(parameters.output_dim_to_input_dim(Size([1, 4]), 1), 8)
