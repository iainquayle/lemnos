#
# NOTE: this only demonstrates how to make a ResNet LIKE search space, not the use of a search controller
# To see the use of a controller, go to the mnist example.
#

from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lemnos.shared import LockedShape, ShapeBound
from lemnos.schema import Schema, SchemaNode, New, PowerGrowth, LinearGrowth, BreedIndices
from lemnos.schema.components import Conv, BatchNorm, ReLU6, Softmax
from lemnos.adapter.torch import generate_source 



head = SchemaNode(ShapeBound(None, None, None), LinearGrowth(10, .2), None, Conv(), ReLU6(), BatchNorm())
head2 = SchemaNode(ShapeBound(None, None, None), LinearGrowth(10, .2), None, Conv(), ReLU6(), BatchNorm())

skip = SchemaNode(ShapeBound(None, None, None), None, None, None, None, BatchNorm())