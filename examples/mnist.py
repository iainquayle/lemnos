from __future__ import annotations

# yes this is hacky af, but it's just for examples
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn

from src.shared import LockedShape, Conformance, ShapeBound, ID
from src.schema import Schema, SchemaNode, New, Existing, PowerGrowth, LinearGrowth, BreedIndices
from src.schema.components import Conv, BatchNorm, ReLU6, Softmax, Full
from src.adapter.torch import TorchEvaluator, generate_source, Adam
from src.control import or_search, MinAvgLossSelector


def main():
	train_data = datasets.MNIST('data', train=True, download=True)
	validation_data = datasets.MNIST('data', train=False, download=True)

	train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
	validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True)
	#oops make sure the output is in tensor

	evaluator = TorchEvaluator(train_loader, validation_loader, 1, nn.CrossEntropyLoss(), None, Adam(0.002), True)
	if (ir := create_schema().compile_ir([LockedShape(1, 28, 28)], BreedIndices(), ID(15))) is not None:
		training_metrics, validation_metrics = evaluator.evaluate(ir)
		print(training_metrics)
			

def create_schema() -> Schema:
	#
	# This schema defines a simple search space, of a model composed of convolutional layers with kernel sizes of either 3 or 5.
	# The first layer is guaranteed to be a 3x3 convolution, all other layers can be any ordering of 3x3 or 5x5 convolutions that results in a 1x1x10 output.
	# The number of channels in the hidden layers will grow following a fractional exponent curve up to 45 channels, with the freedom to vary by 20%.
	#

	start = SchemaNode(ShapeBound(None, None, None), LinearGrowth(10, .2), None, Conv(3), ReLU6(), BatchNorm(), 1, "start")
	conv_3 = SchemaNode(ShapeBound(None, None, None), PowerGrowth(45, .7, .2), None, Conv(3), ReLU6(), BatchNorm(), 1, "conv_3")
	conv_5 = SchemaNode(ShapeBound(None, None, None), PowerGrowth(45, .5, .2), None, Conv(5), ReLU6(), BatchNorm(), 1, "conv_5")
	end = SchemaNode(ShapeBound(10, 1, 1), None, None, Conv(2), Softmax(), None, 1, "end")

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
