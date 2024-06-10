from __future__ import annotations

# yes this is hacky af, but it's just for examples
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

from lemnos.shared import LockedShape, Conformance, ShapeBound, ID
from lemnos.schema import Schema, SchemaNode, New, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax
from lemnos.adapter.torch import TorchEvaluator, generate_source, Adam
from lemnos.control import or_search, AvgEpochLossSelector


#
# This example demonstrates a simple search using pre built components.
# The most important part is the schema definition, everything else is boilerplate that has been abstracted away but can be replaced when needed.
# Infact, it is encouraged to replace at least the evaluator based on the needs of the goal.
#
# IF YOU ARE READING THIS, it currently doesnt save or output the results
#	as in there isnt a way built in, figuring out where to put it


def main():
	train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	validation_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

	train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
	validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True, pin_memory=True)

	accuracy_func = lambda x, y: torch.mean((x.argmax(dim=1) == y).float()).item()
	evaluator = TorchEvaluator(train_loader, validation_loader, 1, nn.CrossEntropyLoss(), accuracy_func, Adam(0.002), True)

	if (ir := create_schema().compile_ir([LockedShape(1, 28, 28)], BreedIndices(), ID(15))) is not None: #purely for demonstration purposes
		print(generate_source("Example", ir))
	else:
		print("Failed to compile schema")

	exit()
	model_pool = or_search(create_schema(), evaluator, AvgEpochLossSelector(), ID(15), 5, 5) 

def create_schema() -> Schema:
	#
	# This schema defines a simple search space, of a model composed of convolutional layers with kernel sizes of either 3 or 5.
	# The first layer is guaranteed to be a 3x3 convolution, all other layers can be any ordering of 3x3 or 5x5 convolutions that results in a 1x1x10 output.
	# The number of channels in the hidden layers will grow following a fractional exponent curve up to 45 channels, with the freedom to vary by 20%.
	#

	start = SchemaNode(ShapeBound(None, None, None), LinearGrowth(10, .2), None, Conv(3), ReLU6(), BatchNorm(), 1, "start")

	conv_3 = SchemaNode(ShapeBound((10, 45), None, None), PowerGrowth(45, .8, .2), None, Conv(3), ReLU6(), BatchNorm(), 1, "conv_3")
	conv_5 = SchemaNode(ShapeBound(None, (6, None), (6, None)), PowerGrowth(45, .6, .2), None, Conv(5), ReLU6(), BatchNorm(), 1, "conv_5")
	# The convolutions above have been given bounds for the purpose of demonstration, but they are not needed  for this graph to be valid.
	# conv_3 will be bound at 10 to 45 channels, while conv_5 is unbounded and free to grow following the growth function.
	# conv_5 is given the lower bound of 6x6, demonstrating how limit the use of a node based on the necessary output shape.
	#	ie, once the output shape of the nodes layers has reached <= 6x6, conv_5 wont be used.

	end = SchemaNode(ShapeBound(10, 1, 1), None, None, Conv(2), Softmax(), None, 1, "end")

	
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
