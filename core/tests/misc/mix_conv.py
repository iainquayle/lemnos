import torch
from torch import nn, zeros


class ConvMix1d(nn.Module):
		import torch
		def __init__(self, channels_in, channels_out, kernel, stride, padding, dilation, groups):
			super().__init__()
			self.c = self.torch.nn.Conv1d(channels_in, channels_out, kernel, stride, padding, dilation, groups,)
			s = channels_out // groups
			self.indices = self.torch.Tensor([i + j * s for j in range(groups) for i in range(s)]).int()
		def forward(self, x):
			return torch.index_select(self.c(x), 1, self.indices)


conv = ConvMix1d(4, 8, 1, 1, 0, 1, 2)

x = zeros(1, 4, 4)
x = conv(x)

print(x.shape)
