import unittest

from torch import Size, zeros
from torch.nn import Conv1d

from src.pattern.node_parameters import IdentityParameters, ConvParameters 
from src.pattern.commons import Bound, Index, Concat, Add, ConformanceShape

from math import prod

class TestParametersShapeFunctions(unittest.TestCase):
	def setUp(self) -> None:
		self.parameters = IdentityParameters()
		self.base_shape = Size([3, 32])
	def test_shape_in_bounds(self) -> None:
		pass
	def test_get_output_shape(self) -> None:
		pass
	def test_get_output_shape_add(self) -> None:
		pass

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
	def test_concat_conformance_shape(self) -> None:
		merge_method = Concat()
		self.assertEqual(merge_method.get_conformance_shape([self.base_shape], Bound([1, 1], [4, 4])), ConformanceShape(2, Size([32])))
		self.assertEqual(merge_method.get_conformance_shape([Size([2, 2, 2])], Bound([1, 1], [4, 4])), ConformanceShape(2, Size([2])))
	def test_add_conformance_shape(self) -> None:
		merge_method = Add()
		self.assertEqual(merge_method.get_conformance_shape([self.base_shape], Bound([1, 1], [4, 4])), ConformanceShape(2, Size([3, 32])))
		self.assertEqual(merge_method.get_conformance_shape([Size([2, 2, 2])], Bound([1, 1], [4, 4])), ConformanceShape(2, Size([4, 2])))
	def test_concat_src(self) -> None:
		merge_method = Concat()
		self.assertEqual(merge_method.get_merge_src(["a", "b"]), "torch.cat([a, b], dim=1)")
	def test_add_src(self) -> None:
		merge_method = Add()
		self.assertEqual(merge_method.get_merge_src(["a", "b"]), "a + b")

class TestConformanceShape(unittest.TestCase):
	def setUp(self) -> None:
		self.a = ConformanceShape(2, Size([2, 3]))
		self.b = ConformanceShape(2, Size([3]))
		self.c = ConformanceShape(3, Size([2, 3]))
		self.d = ConformanceShape(3, Size([1, 2, 3]))
		self.e = ConformanceShape(2, Size([4, 3]))
		self.f = ConformanceShape(2, Size([3, 3]))
	def test_both_constrained(self) -> None:
		self.assertEqual(ConformanceShape.reduce_collection([self.a, self.a]), self.a)
		self.assertEqual(ConformanceShape.reduce_collection([self.a, self.d]), self.d)
		self.assertIsNone(ConformanceShape.reduce_collection([self.a, self.e]))
	def test_one_constrained(self) -> None:
		self.assertEqual(ConformanceShape.reduce_collection([self.a, self.b]), self.a)
		self.assertEqual(ConformanceShape.reduce_collection([self.a, self.c]), self.d)
		self.assertEqual(ConformanceShape.reduce_collection([self.a, ConformanceShape(2, Size())]), self.a)
		self.assertIsNone(ConformanceShape.reduce_collection([self.a, self.f]))
	def test_none_constrained(self) -> None:
		self.assertEqual(ConformanceShape.reduce_collection([self.b, self.b]), self.b)
		self.assertEqual(ConformanceShape.reduce_collection([self.b, self.c]), self.c)
		self.assertEqual(ConformanceShape.reduce_collection([self.b, ConformanceShape(2, Size())]), self.b)

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
