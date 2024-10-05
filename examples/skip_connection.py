#
# NOTE: this only demonstrates how to make a convolution search space with a skip connection 
# To see the use of a controller, go to the mnist example.
#

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lemnos.shared import LockedShape, ShapeBound
from lemnos.schema import Schema, SchemaNode, New, Existing, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax, Sum 
from lemnos.adapter.torch import standard_generator 



head = SchemaNode((None, None, None), LinearGrowth(4, .2), None, Conv(3, 1), ReLU6())

transform = SchemaNode((None, None, None), None, None, Conv(), ReLU6(), BatchNorm())
skip = SchemaNode(ShapeBound(None, None, None), None, Sum(), None, None, BatchNorm())

downsample = SchemaNode((None, (4, None), (4, None)), PowerGrowth(256, .6, .2), Sum(), Conv(2, 0, 2), ReLU6())

end = SchemaNode(ShapeBound(10, 1, 1), None, None, Conv(4, 0), Softmax(), None)

head.add_group(New(transform, 0), New(skip, 1))

transform.add_group(Existing(skip, 1))
skip.add_group(New(transform, 0), New(skip, 1))
skip.add_group(New(downsample, 1))
downsample.add_group(New(skip, 1), New(transform, 0))
downsample.add_group(New(end, 0))

schema = Schema([head], [end])

if (ir := schema.compile_ir([LockedShape(3, 32, 32)], BreedIndices(), 100)) is not None:
	print(standard_generator.generate_source('Test', ir))
else:
	print("Failed to compile schema")
