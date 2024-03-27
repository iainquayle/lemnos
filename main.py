from __future__ import annotations

from src.shared import LockedShape, ID
from src.schema import Schema, SchemaNode, JoinType, BreedIndices
from src.schema.components import *
from src.adapter import generate_torch_module, get_module

import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split
import pandas as pd

class IMDBDataset(Dataset):
	def __init__(self, path: str):
		csv = pd.read_csv(path)
		self.data = [review[0:] for review in csv["review"]]
		self.labels = [Tensor([1 if sentiment == 'positive' else 0]) for sentiment in csv["sentiment"]]
	def __len__(self) -> int:
		return len(self.data)	
	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		return torch.stack([char_to_class(char) for char in self.data[index]]), self.labels[index]

punctuation = ".,:!?()=/\"' +-*@#$%&"
replace_map = {
	'á': 'a',
	'à': 'a',
	'â': 'a',
	'ä': 'a',
	'é': 'e',
	'è': 'e',
	'ê': 'e',
	'ë': 'e',
	'í': 'i',
	'ì': 'i',
	'î': 'i',
	'ï': 'i',
	'ó': 'o',
	'ò': 'o',
	'ô': 'o',
	'ö': 'o',
	'ú': 'u',
	'ù': 'u',
	'û': 'u',
	'ü': 'u',
	"ç": "c",
	"ñ": "n",
	";": ".",
	"[": "(",
	"{": "(",
	"]": ")",
	"}": ")",
	'<': '(',
	'>': ')',
	'`': "'",
	'_': ' ',
	'\n': ' ',
	'\t': ' ',
	'\\': '/',
	'\r': ' ',
	'|': ' ',
}
CLASS_SIZE = 26 + 10 + len(punctuation) 
def char_to_class(char: str) -> Tensor:
	if len(char) != 1:
		raise ValueError("input must be single char")
	output = torch.zeros(CLASS_SIZE, dtype=torch.float32)
	value = ord(char)
	index = 0
	#if 'A' <= char <= 'Z':
		#output[value - ord('A') + index] = 1.0
		#return output
	#else:
	#	index += 26	
	char = char.lower()
	if char in replace_map:
		char = replace_map[char]
	if 'a' <= char <= 'z':
		output[value - ord('a')] = 1.0
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
	print("Invalid character: " + char)
	return output



print(CLASS_SIZE)

#dataset = IMDBDataset("data/imdb.csv")
#train, test = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

input_length = 2**14
start = SchemaNode(ShapeBound((5, 20), None), Sum(), 
	Conv((0.1, 0.5), 1, 1), None, None)
down_sample = SchemaNode(ShapeBound((16, 256), (1, input_length)), Sum(), 
	Conv((0.20, 1.0), 4, 4), None, None)
end = SchemaNode(ShapeBound(CLASS_SIZE, 1), Sum(), Full((0.1, 10)), None, None)
start.add_group((down_sample, 0, JoinType.NEW))
down_sample.add_group((down_sample, 0, JoinType.NEW))
down_sample.add_group((end, 0, JoinType.NEW))
schema = Schema([start], [end])

#ir = schema.compile_ir([LockedShape(CLASS_SIZE, input_length)], BreedIndices(), ID(300))
#if ir is not None:
#	print(generate_torch_module("M", ir))



