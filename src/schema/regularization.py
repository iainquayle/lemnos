from __future__ import annotations

from abc import ABC as Abstract

class Regularization(Abstract):
	pass
class Dropout(Regularization):
	pass
class BatchNormalization(Regularization):
	pass
