import torch

one_d = torch.tensor(range(0, 8))
two_d = one_d.view(-1, 2)
three_d = one_d.view(2, 2, 2)
#options:
#when convert between channels, shuffle all dims up
#make for easier depthwise convolution creation, using 3d convs
#consider making convs output something with nonly 1 in first dim
#may use higher dimensional convs for all?
#if wanting 1x1 then use Dx1x1 with padding valid
def dim_test_down(x):
	return x.view([-1] + list(x.shape)[2:])
def dim_test_up(x):
	return x.view([1] + list(x.shape))
def dim_test_squish(x):
	return x.view([1, -1] + list(x.shape)[2:])


print(three_d)
print(dim_test_up(three_d))
print(dim_test_down(three_d))
print(dim_test_squish(three_d))