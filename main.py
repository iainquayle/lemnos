from __future__ import annotations

from src.shared import LockedShape, ID
from src.schema import Schema, SchemaNode, JoinType, BreedIndices
from src.schema.components import *
from src.adapter import generate_torch_module, get_module
from src.control.torch_control import Control 

import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split
import pandas as pd

review_length = 2**14
class IMDBDataset(Dataset):
	def __init__(self, path: str):
		csv = pd.read_csv(path)
		self.data = [review[0:] for review in csv["review"]]
		self.labels = [Tensor([1 if sentiment == 'positive' else 0]) for sentiment in csv["sentiment"]]
	def __len__(self) -> int:
		return len(self.data)	
	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		tensor = torch.zeros(CLASS_SIZE, review_length, dtype=torch.float32)
		for i, char in enumerate(self.data[index]):
			if (char_index := char_to_class_index(char)) is not None:
				tensor[char_index][i] = 1.0
		return tensor, self.labels[index]

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
def char_to_class_index(char: str) -> int | None:
	if len(char) != 1:
		raise ValueError("input must be single char")
	#if 'A' <= char <= 'Z':
		#output[value - ord('A') + index] = 1.0
		#return output
	#else:
	#	index += 26	
	if char in replace_map:
		char = replace_map[char]
	char = char.lower()
	value = ord(char)
	offset = 0
	if 'a' <= char <= 'z':
		return value - ord('a')
	else:
		offset += 26
	if '0' <= char <= '9':
		return value - ord('0') + offset 
	else:
		offset += 10
	if (punc_index := punctuation.find(char)) != -1:
		return punc_index + offset 
	#print("Invalid character: " + char)
	return None 



print(CLASS_SIZE)

dataset = IMDBDataset("data/imdb.csv")
train, test = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
del dataset

start = SchemaNode(ShapeBound((5, 15), None), Sum(), 
	Conv((0.1, 0.5), 1, 1), ReLU6(), BatchNormalization())

skip = SchemaNode(ShapeBound(None, (1, review_length)), Sum(), None, None, BatchNormalization())
expand = SchemaNode(ShapeBound((16, 384), None), Sum(), 
	Conv((1, 4), 1, 1), ReLU6(), BatchNormalization())
depthwise = SchemaNode(ShapeBound(None, None), Sum(), 
	Conv(1.0, 7, 1, 1, 3, 1), ReLU6(), BatchNormalization())
shrink = SchemaNode(ShapeBound((16, 384), None), Sum(), 
	Conv((0.25, 1), 1, 1), None, BatchNormalization())

down_sample = SchemaNode(ShapeBound((16, 200), (1, review_length)), Sum(), 
	Conv((0.20, 1.0), 4, 4), ReLU6(), BatchNormalization())

end = SchemaNode(ShapeBound(1, 1), Sum(), Full((0.1, 10)), None, None)

start.add_group((skip, 0, JoinType.NEW))

skip.add_group((expand, 0, JoinType.NEW), (skip, 1, JoinType.NEW))
expand.add_group((depthwise, 0, JoinType.NEW))
depthwise.add_group((shrink, 0, JoinType.NEW))
shrink.add_group((skip, 0, JoinType.EXISTING))

skip.add_group((down_sample, 0, JoinType.NEW))
skip.add_group((end, 0, JoinType.NEW))

down_sample.add_group((skip, 0, JoinType.NEW))
down_sample.add_group((down_sample, 0, JoinType.NEW))
down_sample.add_group((end, 0, JoinType.NEW))

schema = Schema([start], [end])

#ir = schema.compile_ir([LockedShape(CLASS_SIZE, review_length)], BreedIndices(), ID(300))
#if ir is not None:
#	print(generate_torch_module("M", ir))

control = Control(schema, train, test, compile_models=False, max_id=ID(62))
control.search([LockedShape(CLASS_SIZE, review_length)], "", torch.nn.BCEWithLogitsLoss(), workers=4, batch_size=64, model_pool_size=5, training_epochs=20)

