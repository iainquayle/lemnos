from __future__ import annotations

# yes this is hacky af, but it's just for examples
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets

from src.shared import LockedShape, Conformance, ShapeBound
from src.schema import Schema, SchemaNode, New, Existing, PowerGrowth, LinearGrowth
from src.schema.components import Conv, BatchNorm, ReLU6, Softmax
from src.adapter.torch import TorchEvaluator
from src.control import or_search, MinAvgLossSelector


def main():
	pass

def create_schema() -> Schema:
	start = SchemaNode(ShapeBound(1, 28, 28), LinearGrowth(10, .2), None, Conv(), ReLU6(), BatchNorm())
	end = SchemaNode(ShapeBound(1, 10), None, None, None, Softmax(), None)
	return Schema([start], [end])
