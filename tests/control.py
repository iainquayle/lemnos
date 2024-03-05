
import unittest

from src.control import Control, OptimizerType
from src.schema import Schema, SchemaNode, Concat, Conv
from src.shared import ShapeBound, LockedShape, Range

from torch.utils.data import Dataset
from torch.nn import MSELoss
import torch

class TestControl(unittest.TestCase):
	def test_basic_optimization(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 10)), Concat(), Conv(Range(.5, 2)))
		input_shape = LockedShape(1, 1)
		schema = Schema([main], [main])
		class TestDataset(Dataset):
			def __init__(self):
				pass
			def __len__(self):
				return 10 
			def __getitem__(self, index):
				return torch.zeros(1, 1), torch.zeros(1, 1)
		control = Control(schema, TestDataset(), TestDataset())
		control.optimize([input_shape], "", MSELoss())
