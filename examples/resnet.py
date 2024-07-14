#
# NOTE: this only demonstrates how to make a ResNet LIKE search space, not the use of a search controller
# To see the use of a controller, go to the mnist example.
#

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lemnos.shared import LockedShape, ShapeBound
from lemnos.schema import Schema, SchemaNode, New, Existing, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax, Sum, FlexibleConv, DepthwiseGrouping
from lemnos.adapter.torch import generate_source 



head = SchemaNode(ShapeBound(None, None, None), LinearGrowth(4, .2), None, Conv(3, 1), ReLU6(), BatchNorm())
head2 = SchemaNode(ShapeBound(None, None, None), LinearGrowth(2, .2), None, Conv(3, 1), ReLU6(), BatchNorm())

dw_point_squeeze = SchemaNode(ShapeBound(None, None, None), LinearGrowth(.5, .2), None, Conv(), ReLU6(), BatchNorm())
depthwise = SchemaNode(ShapeBound(None, None, None), None, None, FlexibleConv(3, 1, 1, 1, DepthwiseGrouping()), ReLU6(), BatchNorm())
dw_point_expand = SchemaNode(ShapeBound(None, None, None), None, None, Conv(), ReLU6(), BatchNorm())

skip = SchemaNode(ShapeBound(None, None, None), None, Sum(), None, None, BatchNorm())
downsample = SchemaNode(ShapeBound(None, (4, None), (4, None)), PowerGrowth(256, .6, .2), Sum(), Conv(2, 0, 2), ReLU6(), BatchNorm())

end = SchemaNode(ShapeBound(10, 1, 1), None, None, Conv(4, 0), Softmax(), None)

head.add_group(New(head2, 0))
head2.add_group(New(skip, 1), New(dw_point_squeeze, 0))

dw_point_squeeze.add_group(New(depthwise, 0))
depthwise.add_group(New(dw_point_expand, 0))
dw_point_expand.add_group(Existing(skip, 0))
dw_point_expand.add_group(Existing(downsample, 0))

skip.add_group(New(dw_point_squeeze, 0), New(skip, 1))
skip.add_group(New(dw_point_squeeze, 0), New(downsample, 1))
skip.add_group(New(end, 0))

downsample.add_group(New(skip, 1), New(dw_point_squeeze, 0))

schema = Schema([head], [end])


if (ir := schema.compile_ir([LockedShape(3, 32, 32)], BreedIndices(), 100)) is not None:
	print(generate_source('Test', ir))
else:
	print("Failed to compile schema")
