import unittest

from lemnos.schema.components import Conv, ConvTranspose 
from lemnos.shared import LockedShape, OpenShape, ShapeBound, ShapeConformance

#TODO: splits these tests up
class TestConv(unittest.TestCase):
	def test_dimension_forward(self) -> None:
		transform = Conv(kernel=4, stride=2, padding=1, dilation=1)
		self.assertEqual(transform.dimension_forward(LockedShape(1, 8), 1), 4)
		transform = Conv(kernel=3, stride=2, padding=1, dilation=1)
		self.assertEqual(transform.dimension_forward(LockedShape(1, 5), 1), 3)
		#self.assertEqual(transform.output_to_input_dim(3), 6)
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

class TestConvTranspose(unittest.TestCase):
	def test_dimension_forward(self) -> None:
		transform = ConvTranspose(kernel=3, stride=1, padding=1, dilation=1)
		self.assertEqual(transform.dimension_transpose_forward(LockedShape(1, 5), 1), 5)
		transform = ConvTranspose(kernel=3, stride=2, padding=1, dilation=1)
		self.assertEqual(transform.dimension_transpose_forward(LockedShape(1, 3), 1), 5)

class TestFull(unittest.TestCase):
	def test_input_to_output_dim(self) -> None:
		pass
	def test_output_to_input_dim(self) -> None:
		pass
