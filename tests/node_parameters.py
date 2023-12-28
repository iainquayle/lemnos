import unittest
import torch
from torch import Size
from torch.nn import Conv1d, Identity

from src.build_structures.node_parameters import IdentityParameters, ConvParameters 
from src.build_structures.commons import Bound, Index, Concat, Add

class TestParametersShapeFunctions(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.shape_bounds = [Bound(1, 3), Bound(1, 32)]
		self.parameters.merge_method = Concat()
		self.base_shape = Size([3, 32])
	def test_shape_in_bounds(self) -> None:
		self.assertTrue(self.parameters.shape_in_bounds(self.base_shape))
		self.assertFalse(self.parameters.shape_in_bounds(Size([3, 33])))
		self.assertFalse(self.parameters.shape_in_bounds(Size([4, 32])))
	def test_get_output_shape(self) -> None:
		pass
	def test_get_output_shape_add(self) -> None:
		self.parameters.merge_method = Add()
		self.assertTrue(self.parameters.get_output_shape(self.base_shape, [self.base_shape]))
		self.assertFalse(self.parameters.get_output_shape(self.base_shape, [Size([3, 33])]))
		self.assertFalse(self.parameters.get_output_shape(self.base_shape, [Size([4, 32])]))

class TestMergeMethod(unittest.TestCase):
	def setUp(self) -> None:
		self.base_shape = Size([3, 32])
		self.none_valid = Size([self.base_shape[0] + 1, self.base_shape[1] + 1])
		self.concat_valid = Size([self.base_shape[0] + 1, self.base_shape[1]])
	def test_concat_validation(self) -> None:
		merge_method = Concat()
		self.assertTrue(merge_method.validate_shapes([self.base_shape, self.concat_valid]))
		self.assertFalse(merge_method.validate_shapes([self.base_shape, self.none_valid]))
	def test_add_validation(self) -> None:
		merge_method = Add()
		self.assertTrue(merge_method.validate_shapes([self.base_shape, self.base_shape]))
		self.assertFalse(merge_method.validate_shapes([self.base_shape, self.concat_valid]))
	def test_concat_string(self) -> None:
		merge_method = Concat()
		self.assertEqual(merge_method.get_merge_string(["a", "b"]), "torch.cat([a, b], dim=1)")
	def test_add_string(self) -> None:
		merge_method = Add()
		self.assertEqual(merge_method.get_merge_string(["a", "b"]), "a + b")

class TestIdentityParemeters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.shape_bounds = [Bound(1, 3), Bound(1, 32)]
		self.parameters.merge_method = Concat()
		self.base_shape = Size([3, 32])
	def test_transform_string(self) -> None:
		self.assertEqual(self.parameters.get_transform_string(self.base_shape, Index()), "Identity()")
	def test_transform_acquisition(self) -> None:
		transform_from_str = eval(self.parameters.get_transform_string(self.base_shape, Index()))
		transform = Identity()
		self.assertEqual(transform_from_str(torch.zeros(self.base_shape)).shape, transform(torch.zeros(self.base_shape)).shape)

class TestConvParameters(unittest.TestCase):
	def setUp(self) -> None:
		self.base_shape = Size([3, 32])
		self.parameters = ConvParameters(stride=2)
	def test_transform_string(self) -> None:
		transform_from_str = eval(self.parameters.get_transform_string(self.base_shape, Index()))
		transform = Conv1d(3, 3, kernel_size=1, stride=2, dilation=1, padding=1)
		self.assertEqual(transform_from_str(torch.zeros(self.base_shape)).shape, transform(torch.zeros(self.base_shape)).shape)
