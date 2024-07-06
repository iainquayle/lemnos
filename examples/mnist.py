from __future__ import annotations

# yes this is hacky af, but it's just for examples
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

from lemnos.shared import LockedShape, ShapeBound
from lemnos.schema import Schema, SchemaNode, New, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax
from lemnos.adapter.torch import TorchEvaluator, generate_source, Adam
from lemnos.control import or_search, AvgLossWindowSelector 


#
# This example demonstrates how to build a schem and how the prebuilt pieces work together, it does not demonstrate the effectiveness of the search.
# The most important part is the schema definition, everything else is boilerplate that has been abstracted away but can be replaced when needed.
# Infact, it is encouraged to replace at least the evaluator based on the needs of the goal.
#


def main():
	train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	validation_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

	train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=1)
	validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=1)

	accuracy_func = lambda x, y: (x.argmax(dim=1) == y).float().sum().item()
	evaluator = TorchEvaluator(train_loader, validation_loader, 2, nn.CrossEntropyLoss(), accuracy_func, Adam(0.002), None, True)

	if (ir := create_schema().compile_ir([LockedShape(1, 28, 28)], BreedIndices(), 15)) is not None: #purely for demonstration purposes
		print(generate_source("Example", ir))
	else:
		print("Failed to compile schema")

	model_pool = or_search(create_schema(), evaluator, AvgLossWindowSelector(10000), 15, 3, 3) 
	# model pool holds the final internal chosen models, though this may be changed in the future to return a larger history of models. 

	for i in range(len(model_pool)):
		print(f"Model {i}:")
		print(f"Training Metrics: {model_pool[i][1]}")
		print(f"Validation Metrics: {model_pool[i][2]}\n")

def create_schema() -> Schema:
	#
	# This schema defines a simple search space, of a model composed of convolutional layers with kernel sizes of either 3 or 5.
	# The first layer is guaranteed to be a 3x3 convolution, all other layers can be any ordering of 3x3 or 5x5 convolutions that results in a 1x1x10 output.
	# The number of channels in the hidden layers will grow following a fractional exponent curve up to 32 channels, with the freedom to vary by 20%.
	#

	start = SchemaNode(ShapeBound(None, None, None), LinearGrowth(6, .3), None, Conv(3), ReLU6(), BatchNorm(), "start")

	conv_3 = SchemaNode(ShapeBound((6, 32), None, None), PowerGrowth(32, .8, .2), None, Conv(3), ReLU6(), BatchNorm(), "conv_3")
	conv_5 = SchemaNode(ShapeBound(None, (4, None), (4, None)), PowerGrowth(32, .6, .2), None, Conv(5), ReLU6(), BatchNorm(), "conv_5")
	# The convolutions above have been given bounds for the purpose of demonstration, but they are not needed  for this graph to be valid.
	# conv_3 will be bound at 6 to 32 channels, while conv_5 is unbounded and free to grow following the growth function.
	# conv_5 is given the lower bound of 4x4, demonstrating how limit the use of a node based on the necessary output shape.
	#	ie, once the output shape of the nodes layers has reached <= 4x4, conv_5 wont be used.

	end = SchemaNode(ShapeBound(10, 1, 1), None, None, Conv(2), Softmax(), None, "end")

	
	# Note: Groups can have more than one transition.
	start.add_group(New(conv_3, 0))
	start.add_group(New(conv_5, 0))

	conv_3.add_group(New(conv_3, 0))
	conv_3.add_group(New(conv_5, 0))

	conv_5.add_group(New(conv_3, 0))
	conv_5.add_group(New(conv_5, 0))

	conv_3.add_group(New(end, 0))
	conv_5.add_group(New(end, 0))

	return Schema([start], [end])

main()
