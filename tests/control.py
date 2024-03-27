
import unittest

from src.control.torch_control import  Control
from src.schema import Schema, SchemaNode 
from src.schema.components import *
from src.shared import ShapeBound, LockedShape

from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss 
import torch

class TestControl(unittest.TestCase):
	def test_basic_optimization(self):
		main = SchemaNode(ShapeBound((1, 1), (1, 10)), None, None, Conv())
		input_shape = LockedShape(1, 1)
		schema = Schema([main], [main])
		class TestDataset(Dataset):
			def __init__(self):
				pass
			def __len__(self):
				return 10 
			def __getitem__(self, index):
				return torch.zeros(1), torch.zeros(1)
		control = Control(schema, TestDataset(), TestDataset(), compile_models=False)
		control.search([input_shape], "", BCEWithLogitsLoss())
