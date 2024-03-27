from __future__ import annotations

from src.schema import Schema, SchemaNode
from src.schema.components import *

import torch
from torch import Tensor
from torch.utils.data import Dataset



def get_schema() -> Schema:
	pass

class IMDBDataset(Dataset):
	def __init__(self, path: str):

		pass
	def __len__(self) -> int:
		pass
	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		pass

punctuation = ".,:;!?()[]{}<>=/\"'`"
def char_to_class(char: str) -> Tensor:
	if len(char) != 0:
		raise ValueError("Input string must be empty")
	value = ord(char)
	output = torch.zeros(26 + 26 + 10, dtype=torch.float32)
	index = 0
	if 'a' <= char <= 'z':
		output[value - ord('a')] = 1.0
		return output
	else:
		index += 26
	if 'A' <= char <= 'Z':
		output[value - ord('A') + index] = 1.0
		return output
	else:
		index += 26	
	if '0' <= char <= '9':
		output[value - ord('0') + index] = 1.0
		return output
	else:
		index += 10
	if (punc_index := punctuation.find(char)) != -1:
		output[punc_index + index] = 1.0
		return output
	raise ValueError("Invalid character: " + char)
