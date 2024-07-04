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
from lemnos.schema.components import Conv, BatchNorm, Softmax, ReLU6, SiLU, Sum, Full, LayerNorm, ChannelDropout, Dropout, InputSqrtBase2Grouping, DepthwiseGrouping
from lemnos.adapter.torch import TorchEvaluator, generate_source, Adam, SGD, StepLR 
from lemnos.control import or_search, AvgLossWindowSelector 

def main():
	if (ir := model_1().compile_ir([LockedShape(3, 32, 32)], BreedIndices(), 70)) is not None: #purely for demonstration purposes
		print(generate_source("Example", ir))
		exit()
		#train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=.5), transforms.RandomErasing(p=.4, scale=(.02, .2)), transforms.RandomVerticalFlip(p=.5)])
		train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=.5)])

		train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
		validation_data = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())

		train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=16)
		validation_loader = DataLoader(validation_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=16)

		accuracy_func = lambda x, y: (x.argmax(dim=1) == y).float().sum().item()
		evaluator = TorchEvaluator(train_loader, validation_loader, 10, nn.CrossEntropyLoss(), accuracy_func, Adam(0.001, 0.001), StepLR(2, 0.5), True)

		train_metrics, validation_metrics = evaluator.evaluate(ir)
		#model_pool = or_search(model_1(), evaluator, AvgLossWindowSelector(1024), 80, 3, 3) 
	else:
		print("Failed to compile schema")


def model_1() -> Schema:
	head_1 = SchemaNode(ShapeBound(48, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())
	head_2 = SchemaNode(ShapeBound(128, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())

	head_1.add_group(New(head_2, 0))

	accume = SchemaNode(ShapeBound(None, (4, None), (4, None)), None, Sum(), None, None, BatchNorm() , debug_name="accume")
	skip = SchemaNode(ShapeBound(None, None, None), None, None, None, None, ChannelDropout(.1), debug_name="skip")
	downsample = SchemaNode(ShapeBound(None, (2, None), (2, None)), PowerGrowth(384, .7, .0), Sum(), Conv(2, 0, 2, 1, InputSqrtBase2Grouping(), mix_groups=True), SiLU(), BatchNorm(), debug_name="downsample")
	dw_3_point = SchemaNode(ShapeBound(None, None, None), LinearGrowth(2, .0), None, Conv(groups=InputSqrtBase2Grouping(), mix_groups=True), ReLU6(), BatchNorm(), debug_name="dw_3_point")
	depthwise_3 = SchemaNode(ShapeBound(None, None, None), None, None, Conv(3, 1, 1, 1, DepthwiseGrouping()), ReLU6(), BatchNorm(), debug_name="depthwise_3")
	dw_collect = SchemaNode(ShapeBound(None, None, None), None, None, Conv(groups=1, mix_groups=True), None, BatchNorm(), debug_name="dw_collect")

	tail_1 = SchemaNode(ShapeBound(256, 1), None, None, Conv(2, 0), ReLU6(), BatchNorm(), debug_name="tail_1")
	tail_2 = SchemaNode(ShapeBound(10, 1), None, None, Full(), Softmax(), None)

	head_2.add_group(New(skip, 1), New(dw_3_point, 0))

	dw_3_point.add_group(New(depthwise_3, 0))
	depthwise_3.add_group(New(dw_collect, 2))
	dw_collect.add_group(Existing(accume, 0))
	dw_collect.add_group(Existing(downsample, 0))
	skip.add_group(New(downsample, 3))
	skip.add_group(New(accume, 3))

	downsample.add_group(New(skip, 1), New(dw_3_point, 0))
	accume.add_group(New(skip, 1), New(dw_3_point, 0))

	downsample.add_group(New(tail_1, 1))

	tail_1.add_group(New(tail_2, 1))

	return Schema([head_1], [tail_2])

def model_2() -> Schema:
	conv_1_1 = SchemaNode(ShapeBound(32, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())
	conv_1_2 = SchemaNode(ShapeBound(32, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())

	downsample_1 = SchemaNode(ShapeBound(32, None, None), None, None, Conv(2, 0, 2, 1), ReLU6(), Dropout(.2))

	conv_2_1 = SchemaNode(ShapeBound(64, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())
	conv_2_2 = SchemaNode(ShapeBound(64, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())

	downsample_2 = SchemaNode(ShapeBound(64, None, None), None, None, Conv(2, 0, 2, 1), ReLU6(), Dropout(.2))
	
	conv_3_1 = SchemaNode(ShapeBound(128, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())
	conv_3_2 = SchemaNode(ShapeBound(128, None, None), None, None, Conv(3, 1), ReLU6(), BatchNorm())

	downsample_3 = SchemaNode(ShapeBound(128, None, None), None, None, Conv(2, 0, 2, 1), ReLU6(), Dropout(.2))

	full = SchemaNode(ShapeBound(128), None, None, Full(), ReLU6(), Dropout(.2))

	end = SchemaNode(ShapeBound(10), None, None, Full(), Softmax(), None)

	conv_1_1.add_group(New(conv_1_2, 0))
	conv_1_2.add_group(New(downsample_1, 0))
	downsample_1.add_group(New(conv_2_1, 0))
	conv_2_1.add_group(New(conv_2_2, 0))
	conv_2_2.add_group(New(downsample_2, 0))
	downsample_2.add_group(New(conv_3_1, 0))
	conv_3_1.add_group(New(conv_3_2, 0))
	conv_3_2.add_group(New(downsample_3, 0))
	downsample_3.add_group(New(full, 0))
	full.add_group(New(end, 0))

	return Schema([conv_1_1], [end])

main()
