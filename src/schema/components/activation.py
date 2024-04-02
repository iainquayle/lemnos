from __future__ import annotations

from abc import ABC as Abstract 

class Activation(Abstract):
	pass

class ReLU(Activation):
	pass
class ReLU6(Activation):
	pass
class Softmax(Activation):
	pass
class Sigmoid(Activation):
	pass
class SiLU(Activation):
	pass
