from torch import Tensor, Size
from torch.nn import Module
import torch.nn as nn

def get_features_shape(x):
	return x.shape[1:]
def get_batch_norm(shape):
	if len(shape) == 1:
		return nn.BatchNorm1d(shape.numel())
	elif len(shape) == 2:
		return nn.BatchNorm2d(shape.numel())
def mould_features(x, shape):
    return x.view([x.shape[0]] + list(shape))
def change_shape_dimension(shape, diff=1):
	return [-1] + list(shape)[1+diff:] if diff >= 0 else [1] * -diff + list(shape)
def squish_shape(shape, diff=1):
	return [1] * diff + [-1] + list(shape)[2:] 
