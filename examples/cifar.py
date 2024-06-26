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
from lemnos.schema import Schema, SchemaNode, New, Existing, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax, SiLU, Sum, Concat, GroupType
from lemnos.adapter.torch import TorchEvaluator, generate_source, Adam
from lemnos.control import or_search, AvgLossWindowSelector 

def main():
	if (ir := create_schema().compile_ir([LockedShape(3, 32, 32)], BreedIndices(), 40)) is not None: #purely for demonstration purposes
		print(generate_source("Example", ir))
	else:
		print("Failed to compile schema")

	train_data = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
	validation_data = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())

	train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=16)
	validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=16)

	accuracy_func = lambda x, y: (x.argmax(dim=1) == y).float().sum().item()
	evaluator = TorchEvaluator(train_loader, validation_loader, 2, nn.CrossEntropyLoss(), accuracy_func, Adam(0.002), True)


	model_pool = or_search(create_schema(), evaluator, AvgLossWindowSelector(4096), 40, 3, 3) 
	# model pool holds the final internal chosen models, though this may be changed in the future to return a larger history of models. 

	for i in range(len(model_pool)):
		print(f"Model {i}:")
		print(f"Training Metrics: {model_pool[i][1]}")
		print(f"Validation Metrics: {model_pool[i][2]}\n")

def create_schema() -> Schema:
	groups = 4

	head_1 = SchemaNode(ShapeBound(32, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())
	head_2 = SchemaNode(ShapeBound(64, None, None), None, None, Conv(2, 1), ReLU6(), BatchNorm())

	head_1.add_group(New(head_2, 0))

	skip = SchemaNode(ShapeBound(None, None, None), None, Sum(), None, None, BatchNorm())
	downsample = SchemaNode(ShapeBound(None, (2, None), (2, None)), PowerGrowth(256, .8, .0), Sum(), Conv(2, 0, 2, 1, groups, mix_groups=True), SiLU(), BatchNorm())

	dw_3_point = SchemaNode(ShapeBound(None, None, None), LinearGrowth(2, .0), None, Conv(groups=groups, mix_groups=True), ReLU6(), BatchNorm())
	depthwise_3 = SchemaNode(ShapeBound(None, None, None), None, None, Conv(3, 1, 1, 1, GroupType.DEPTHWISE), ReLU6(), BatchNorm())
	dw_collect = SchemaNode(ShapeBound(None, None, None), None, None, Conv(groups=groups, mix_groups=True), None, BatchNorm())

	end = SchemaNode(ShapeBound(10, 1), None, None, Conv(2, 0), Softmax(), None)

	head_2.add_group(New(skip, 3), New(dw_3_point, 0))

	dw_3_point.add_group(New(depthwise_3, 0))
	depthwise_3.add_group(New(dw_collect, 2))
	dw_collect.add_group(Existing(skip, 0))

	skip.add_group(New(skip, 3), New(dw_3_point, 0))

	dw_collect.add_group(Existing(downsample, 0))
	skip.add_group(New(downsample, 3), New(dw_3_point, 0))

	downsample.add_group(New(skip, 3), New(dw_3_point, 0))

	skip.add_group(New(end, 0))

	return Schema([head_1], [end])

main()
