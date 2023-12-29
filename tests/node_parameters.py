import unittest
import torch
from torch import Size
from torch.nn import Conv1d, Identity

from src.build_structures.node_parameters import IdentityParameters, ConvParameters 
from src.build_structures.commons import Bound, Index, Concat, Add

from math import prod

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
		pass

class TestNodeParameters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.merge_method = Concat()
		self.base_shape = Size([2, 4])
	def test_get_mould_shape_same_dim(self) -> None:
		self.parameters.shape_bounds = [Bound(1, 4), Bound(1, 4)]
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape, self.base_shape]), Size([4, 4]))
	def test_get_mould_shape_up_dim(self) -> None:
		self.parameters.shape_bounds = [Bound(1, 4), Bound(1, 4), Bound(1, 4)]
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape]), Size([1, 2, 4]))
	def test_get_mould_shape_down_dim(self) -> None:
		self.parameters.shape_bounds = [Bound(1, 4)]
		self.assertEqual(self.parameters.get_mould_shape([self.base_shape]), Size([8]))

class TestMergeMethod(unittest.TestCase):
	def setUp(self) -> None:
		self.base_shape = Size([3, 32])
		self.none_valid = Size([self.base_shape[0] + 1, self.base_shape[1] + 1])
		self.concat_valid = Size([self.base_shape[0] + 1, self.base_shape[1]])
	def test_concat_total_merged_size(self) -> None:
		merge_method = Concat()
		self.assertEqual(merge_method.get_total_merged_size([self.base_shape, self.base_shape]), prod(self.base_shape) * 2)
	def test_add_total_merged_size(self) -> None:
		merge_method = Add()
		self.assertEqual(merge_method.get_total_merged_size([self.base_shape, self.base_shape]), prod(self.base_shape))
	def test_concat_src(self) -> None:
		merge_method = Concat()
		self.assertEqual(merge_method.get_merge_src(["a", "b"]), "torch.cat([a, b], dim=1)")
	def test_add_src(self) -> None:
		merge_method = Add()
		self.assertEqual(merge_method.get_merge_src(["a", "b"]), "a + b")

class TestIdentityParemeters(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.parameters.shape_bounds = [Bound(1, 3), Bound(1, 32)]
		self.parameters.merge_method = Concat()
		self.base_shape = Size([3, 32])
	def test_transform_src(self) -> None:
		self.assertEqual(self.parameters.get_transform_src(self.base_shape, self.base_shape), "Identity()")
	def test_transform(self) -> None:
		pass

class TestConvParameters(unittest.TestCase):
	def setUp(self) -> None:
		self.base_shape = Size([3, 32])
		self.parameters = ConvParameters(stride=2)
	def test_transform_src(self) -> None:
		pass
