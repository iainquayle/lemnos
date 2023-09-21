import unittest

from torch.nn.modules import module

from src.kontrol.node_info import NodeInfo, BasicConvInfo, IdentityInfo, Index, Bound
from torch import nn, Size

class TestBuild(unittest.TestCase):
	def test_size_from_index(self):
		bound = Bound(-1, 1)	
		self.assertAlmostEqual(0.0, bound.from_ratio(ratio=0.5), delta=0.01)	
		bound = Bound(0, 1)
		self.assertAlmostEqual(0.5, bound.from_ratio(ratio=0.5), delta=0.01)	
	def test_basic_conv_1d(self):
		base_info = NodeInfo([Bound(1, 10), Bound(1, 10), Bound(1, 10),], Bound(1.0, 2.0), [nn.ReLU()])
		info = BasicConvInfo(base_info, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1), padding=(0, 0))
		module = info.get_function(Size([2, 2, 2, 2]), Index())
		module2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1), padding=(0, 0))
		self.assertEqual(str(module), str(module2))
	def test_basic_conv_2d(self):
		base_info = NodeInfo([Bound(1, 10), Bound(1, 10), Bound(1, 10),], Bound(1.0, 2.0), [nn.ReLU()])
		info = BasicConvInfo(base_info, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1), padding=(0, 0))
		module = info.get_function(Size([2, 2, 2, 2]), Index())
		module2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1), padding=(0, 0))
		self.assertEqual(str(module), str(module2))