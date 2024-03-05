
import unittest

from src.control import Control
from src.schema import Schema, SchemaNode, Concat
from src.shared import ShapeBound, LockedShape

from torch.utils.data import Dataset
from torch.optim import Adam 
from torch.nn import MSELoss

class TestControl(unittest.TestCase):
	def test_basic_optimization(self):
		main = SchemaNode(ShapeBound((1, 10)), Concat())
		input_shape = LockedShape(5)
		schema = Schema([main], [main])
		class TestDataset(Dataset):
			def __init__(self):
				pass
			def __len__(self):
				return 0
			def __getitem__(self, index):
				return 0
		control = Control(schema, TestDataset(), TestDataset())
		control.optimize([input_shape], "", 1, Adam, MSELoss(), 1)
