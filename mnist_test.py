from __future__ import annotations

from src.shared import LockedShape, ShapeBound, ID
from src.schema import Schema, SchemaNode, JoinType, BreedIndices
from src.schema.components import *
from src.adapter import generate_torch_module, get_module
from src.control.torch_control import Control 

import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split


train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())




loop = SchemaNode(ShapeBound((1, 64), (4, 28), (4, 28)), Sum(), 
	Conv((1.2, 6), 5), ReLU6(), BatchNormalization())
hidden_fc = SchemaNode(ShapeBound(64, 4, 4), Sum(), Full(1), ReLU6())
end = SchemaNode(ShapeBound(10), Sum(), Full((.5, 2)), Softmax())

loop.add_group((loop, 0, JoinType.NEW))
loop.add_group((hidden_fc, 0, JoinType.NEW))
hidden_fc.add_group((end, 0, JoinType.NEW))

schema = Schema([loop], [end])

input_shape = LockedShape(1, 28, 28)
if (ir := schema.compile_ir([input_shape], BreedIndices(), ID(10))) is not None:
	print(generate_torch_module("M", ir))


control = Control(schema, train, test, compile_models=False, max_id=ID(10))
control.search([input_shape], "", torch.nn.CrossEntropyLoss(), workers=2, batch_size=16, model_pool_size=5, training_epochs=2)
