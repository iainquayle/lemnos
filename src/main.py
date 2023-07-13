import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

#TODO: pass back model dimensions, 
# depending on the dimensions of the input, and some other parameters, modules can be created?
#make own sequential wrapper which can take in some dimension, 
# these can then be turned into real sequentials, and when parallel merely a list of these can easily be combined

#course 1
#make dag of modules
# each node carries native module, parent count, and children
# generate internal module using builder pattern mechanics
#  allows for manipulation during creation

#course 2
#make rule set that operates on list of modules, will have to create warpper for certain modules like concat

#course 3
#course 2 but create more sophisticated wrappers, that hold relevant information, like input shape, parents, children

print("Testing model_make.py")
test_tensor = torch.rand(1, 1, 28, 28)
print(test_tensor.shape)
test_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same')
test_tensor = test_conv(test_tensor)
#clones rather than copies
test_tensor2 = test_tensor
print(test_tensor.shape)
test_conv = nn.Conv2d(in_channels=test_tensor.shape[1], out_channels=16, kernel_size=2, stride=2, padding='valid')
test_tensor = test_conv(test_tensor)
print(test_tensor.shape)
print(test_tensor2.shape)

