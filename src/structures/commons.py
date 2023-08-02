import torch
from torch import Tensor, Size
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Any
from copy import copy, deepcopy
from math import prod
from enum import Enum

identity = lambda x: x
class Identity(Module):
	def __init__(self) -> None:
		super().__init__()
	def forward(self, x: Tensor) -> Tensor:
		return x
class MergeMethod(Enum):
	CONCAT = 'concat'
	ADD = 'add'
	SINGLE = 'single'
	LIST = 'list'
	def get_function(self) -> Callable[[List[Tensor]], Tensor | List[Tensor]]:
		if self == MergeMethod.CONCAT:
			return lambda x: torch.cat(x, dim=1)
		elif self == MergeMethod.ADD:
			return lambda x: sum(x)
		elif self == MergeMethod.SINGLE:
			return lambda x: x[0]
		elif self == MergeMethod.LIST:
			return identity 
		else:
			return None
def get_features_shape(x):
	return x.shape[1:]
def get_batch_norm(shape):
	if len(shape) == 1:
		return nn.BatchNorm1d(shape.numel())
	elif len(shape) == 2:
		return nn.BatchNorm2d(shape.numel())
def mould_features(x, shape):
    return x.view([x.shape[0]] + list(shape))
@staticmethod
def change_shape_dimension(shape, diff=1):
	return [-1] + list(shape)[1+diff:] if diff >= 0 else [1] * -diff + list(shape)
@staticmethod
def squish_shape(shape, diff=1):
	return [1] * diff + [-1] + list(shape)[2:] 