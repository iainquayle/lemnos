import torch
import torch.nn as nn

tensor = torch.rand(1, 2, 2)
print(tensor)
c1 = nn.Conv2d(1, 1, kernel_size=2, padding=0)
print(c1(tensor))
#results in -1 size
c2 = nn.Conv2d(1, 1, kernel_size=2, padding=1)
print(c2(tensor))
#results in +1 size
c3 = nn.Conv2d(1, 1, kernel_size=2, padding='same')
print(c3(tensor))
#required padding the processing