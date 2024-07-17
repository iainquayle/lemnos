import unittest

from torch import zeros
from torch.nn import Conv1d

from lemnos.schema.components import Conv, FlexibleConv
from lemnos.shared import LockedShape, OpenShape, ShapeBound, ShapeConformance

#TODO: splits these tests up
class TestConv(unittest.TestCase):
	def test_input_to_output_dim(self) -> None:
		input = zeros(1, 1, 9)
		shape = LockedShape(*input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, bias=False)
		transform = Conv(kernel=3, stride=1, padding=0, dilation=1)
		self.assertEqual(transform.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 8)
		shape = LockedShape(*input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
		transform = Conv(kernel=2, stride=2, padding=0, dilation=1)
		self.assertEqual(transform.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 9)
		shape = LockedShape(*input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, dilation=2, bias=False)
		transform = Conv(kernel=2, stride=1, padding=1, dilation=2)
		self.assertEqual(transform.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		input = zeros(1, 1, 8)
		shape = LockedShape(*input.size()[1:])
		reference = Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=2, dilation=4, bias=False)
		transform = Conv(kernel=2, stride=1, padding=2, dilation=4)
		self.assertEqual(transform.input_dim_to_output_dim(shape, 1), reference(input).shape[2])
		#print(reference(input).shape[2])
	def test_output_to_input_dim(self) -> None:
		pass
	def test_mould_output_shape_valid_upper(self) -> None:
		transform = Conv(kernel=2, stride=2, padding=0, dilation=1)
		shape = LockedShape(2, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(2, 3), 1), ShapeBound((2, 2), (1, 8)), 1)) is not None:
			self.assertEqual(shape, LockedShape(2, 3))
		transform = Conv(kernel=2, stride=2, padding=0, dilation=1)
		shape = LockedShape(2, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 1), ShapeBound((2, 2), (1, 8)), 1)) is not None:
			self.assertEqual(shape, LockedShape(2, 3))
		shape = LockedShape(2, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(LockedShape(2, 3), 1), ShapeBound((1, 1000), (1, 8)), 1)) is not None:
			self.assertEqual(shape, LockedShape(2, 3))
	def test_mould_output_shape_valid_lower(self) -> None:
		transform = Conv(kernel=2, stride=2, padding=0, dilation=1, groups=2)
		shape = LockedShape(2, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 1), ShapeBound((2, 2), (1, 8)), .5)) is not None:
			self.assertEqual(shape, LockedShape(2, 3))
		shape = LockedShape(4, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 1), ShapeBound((1, 4), (1, 8)), .5)) is not None:
			self.assertEqual(shape, LockedShape(2, 3))
		shape = LockedShape(2, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 1), ShapeBound((1, 4), (1, 8)), 2)) is not None:
			self.assertEqual(shape, LockedShape(4, 3))
		shape = LockedShape(4, 6)
		if (shape := transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 3), ShapeBound((1, 8), (1, 8)), 2)) is not None:
			self.assertEqual(shape, LockedShape(6, 3))
	def test_output_shape_invalid(self) -> None:
		transform = Conv(kernel=2, stride=2, padding=0, dilation=1, groups=2)
		shape = LockedShape(2, 6)
		self.assertIsNone(transform.get_output_shape(shape, ShapeConformance(LockedShape(3, 3), 1), ShapeBound((2, 2), (1, 8)), 1))
		shape = LockedShape(3, 6)
		self.assertIsNone(transform.get_output_shape(shape, ShapeConformance(OpenShape(3), 1), ShapeBound((2, 2), (1, 8)), 1))
	def test_flex_conv_splitting(self) -> None:
		conv = FlexibleConv(groups=3)
		conv_splits, mix_indices = conv.get_conv_splits_and_mix_indices(LockedShape(8, 1), LockedShape(8, 1))
		self.assertEqual(len(conv_splits), 2)
		self.assertEqual(conv_splits[0][0], 2)
		self.assertEqual(conv_splits[0][2], 1)
		self.assertEqual(len(mix_indices), 8)

class TestFull(unittest.TestCase):
	def test_input_to_output_dim(self) -> None:
		pass
	def test_output_to_input_dim(self) -> None:
		pass
