from __future__ import annotations

from abc import ABC as Abstract, abstractmethod

class Activation(Abstract):
	@abstractmethod
	def get_activation_src(self, register_in: str) -> str:
		pass
class ReLU(Activation):
	def get_activation_src(self, register_in: str) -> str:
		return "tf.nn.relu"
	pass
