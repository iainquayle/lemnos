from __future__ import annotations

from src.shared import LockedShape, ID
from src.schema import Schema, SchemaNode, JoinType, BreedIndices, PowerGrowth, LinearGrowth
from src.schema.components import *
from src.adapter import generate_torch_module, get_module
from src.control.torch_control import Control 

import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split
import pandas as pd

import re

TWEET_LENGTH = 257 
def _clean_tweet(tweet: str) -> str:
	tweet = re.sub("@[a-zA-Z_0-9]+", "@someone", tweet)
	tweet = re.sub("https?://[a-zA-Z0-9./]+", "http://link", tweet)
	return tweet[:TWEET_LENGTH]
class TweetDataset(Dataset):
	def __init__(self, path: str):
		csv = pd.read_csv(path, encoding="ISO-8859-1")
		self.data = [_clean_tweet(tweet) for tweet in csv["text"]]
		self.labels = [Tensor([1 if sentiment == 4 else 0]) for sentiment in csv["sentiment"]]
		print(self.labels[0])
	def __len__(self) -> int:
		return len(self.data)
	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		tensor = torch.zeros(CLASS_SIZE, TWEET_LENGTH, dtype=torch.float32)
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


def get_schema_a():
	start = SchemaNode(ShapeBound((5, 15), None), 
		LinearGrowth(1/5, .5),
		None, 
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm())
	skip = SchemaNode(ShapeBound(None, (1, None)), 
		None,
		Sum(), 
		None, 
		None, 
		BatchNorm())
	expand = SchemaNode(ShapeBound((32, 384), None), 
		LinearGrowth(1/5, .5),
		None, 
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm())
	depthwise = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(), 
		Conv(7, 1, 1, 2, 1), 
		ReLU6(), 
		BatchNorm())
	shrink = SchemaNode( ShapeBound((32, 256), None), 
		LinearGrowth(1/5, .5),
		Sum(), 
		Conv(1, 1), 
		None, 
		BatchNorm())
	down_sample = SchemaNode(ShapeBound((16, 256), (1, None)), 
		PowerGrowth(220, .5, .25),
		Sum(), 
		Conv(2, 2), 
		SiLU(), 
		BatchNorm())
	end = SchemaNode(ShapeBound(1, 1), 
		None,
		Sum(), 
		Full(), None, None)
	#start.add_group((expand, 0, JoinType.NEW), (skip, 1, JoinType.NEW))
	start.add_group((skip, 1, JoinType.NEW))
	skip.add_group((expand, 0, JoinType.NEW), (skip, 1, JoinType.NEW))
	expand.add_group((depthwise, 0, JoinType.NEW))
	depthwise.add_group((shrink, 0, JoinType.NEW))
	shrink.add_group((skip, 0, JoinType.EXISTING))
	skip.add_group((down_sample, 0, JoinType.NEW))
	skip.add_group((end, 0, JoinType.NEW))
	#down_sample.add_group((expand, 0, JoinType.NEW), (skip, 1, JoinType.NEW))
	down_sample.add_group((skip, 1, JoinType.NEW))
	down_sample.add_group((down_sample, 0, JoinType.NEW))
	down_sample.add_group((end, 0, JoinType.NEW))

	return Schema([start], [end])

def get_schema_b():
	#two options, give each depthwise a seperate pointwise to allow for different dimensions
	#	or as is now, have them all pull directly from the skip and have a fixed size
	embed = SchemaNode(ShapeBound((10, 16), None), 
		LinearGrowth(1/5, .5),
		Sum(), 
		Conv(1, 1), ReLU6(), 
		BatchNorm())
	second = SchemaNode(ShapeBound((40, 64), None), 
		LinearGrowth(4, .3),
		Sum(), 
		Conv(3, 1, 1, 1),
		ReLU6(),
		BatchNorm(), 8, "second")
	skip = SchemaNode(ShapeBound(None, (1, None)), 
		None,
		Sum(), 
		None, 
		None, 
		BatchNorm(), 1, "skip")
	shrink = SchemaNode(ShapeBound((32, 256), None), 
		None,
		Concat(), 
		Conv(1, 1), 
		None, 
		BatchNorm(), 1, "shrink")
	expand_s = SchemaNode(ShapeBound((32, 128), None), 
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), 8, "expand_s")
	expand_m = SchemaNode(ShapeBound((32, 128), None),
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), 8, "expand_m")
	expand_l = SchemaNode(ShapeBound((32, 128), None),
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), 8, "expand_l")
	depthwise_s = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(), 
		Conv(3, 1, 1, 1, 1), 
		ReLU6(), 
		BatchNorm(), 1, "depthwise_s")
	depthwise_m = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(),
		Conv(2, 1, 4, 2, 1), 
		ReLU6(), 
		BatchNorm(), 1, "depthwise_m")
	depthwise_l = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(),
		Conv(2, 1, 6, 3, 1), 
		ReLU6(), 
		BatchNorm(), 1, "depthwise_l")
	down_sample_point = SchemaNode(ShapeBound((32, 256), (1, None)), 
		PowerGrowth(220, .6, .25),
		Sum(),
		Conv(1, 1), 
		SiLU(), 
		BatchNorm(), 8)
	down_sample_depthwise = SchemaNode(ShapeBound(None, (1, None)), 
		None,
		Sum(),
		Conv(2, 2, 1, 0, 1), 
		SiLU(), 
		BatchNorm())
	down_sample = SchemaNode(ShapeBound((16, 256), (1, None)), 
		PowerGrowth(220, .6, .25),
		Sum(), 
		Conv(2, 2), 
		SiLU(), 
		BatchNorm(), 8, "down_sample")
	end = SchemaNode(ShapeBound(1, 1), 
		None,
		Sum(), 
		Full(), 
		None, 
		None)
	embed.add_group((second, 0, JoinType.NEW))
	second.add_group((down_sample, 0, JoinType.NEW))
	#second.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW))
	second.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW), (expand_m, 0, JoinType.NEW), (expand_l, 0, JoinType.NEW))
	#skip.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW))
	skip.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW), (expand_m, 0, JoinType.NEW), (expand_l, 0, JoinType.NEW))
	skip.add_group((down_sample, 0, JoinType.NEW))
	expand_s.add_group((depthwise_s, 0, JoinType.NEW))
	expand_m.add_group((depthwise_m, 1, JoinType.NEW))
	expand_l.add_group((depthwise_l, 1, JoinType.NEW))
	depthwise_s.add_group((shrink, 2, JoinType.NEW))
	depthwise_m.add_group((shrink, 2, JoinType.EXISTING))
	depthwise_l.add_group((shrink, 2, JoinType.EXISTING))
	shrink.add_group((skip, 0, JoinType.EXISTING))
	#something with cat is breaking shit
	down_sample.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW), (expand_m, 0, JoinType.NEW), (expand_l, 0, JoinType.NEW))
	#down_sample.add_group((skip, 3, JoinType.NEW), (expand_s, 0, JoinType.NEW))
	down_sample.add_group((down_sample, 0, JoinType.NEW))
	down_sample.add_group((end, 0, JoinType.NEW))
	skip.add_group((end, 0, JoinType.NEW))
	#down_sample_point.add_group((down_sample_depthwise, 0, JoinType.NEW))
	return Schema([embed], [end])


def get_schema_c():
	shrink_groups = 8
	embed = SchemaNode(ShapeBound((10, 16), None), 
		LinearGrowth(1/5, .5),
		Sum(), 
		Conv(1, 1), ReLU6(), 
		BatchNorm())
	second = SchemaNode(ShapeBound((40, 80), None), 
		LinearGrowth(4, .3),
		Sum(), 
		Conv(2, 1, 1, 0),
		ReLU6(),
		BatchNorm(), shrink_groups, "second")
	skip = SchemaNode(ShapeBound(None, (1, None)), 
		None,
		Sum(), 
		None, 
		None, 
		BatchNorm(), 1, "skip")
	expand_p = SchemaNode(ShapeBound((32, 256), None), 
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), shrink_groups, "expand_s")
	expand_2 = SchemaNode(ShapeBound((32, 256), None),
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), shrink_groups, "expand_m")
	expand_6 = SchemaNode(ShapeBound((32, 256), (6, None)),
		LinearGrowth(3/4, .25),
		Sum(),
		Conv(1, 1), 
		ReLU6(), 
		BatchNorm(), shrink_groups, "expand_l")
	depthwise_2 = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(),
		Conv(2, 1, 2, 1, 1), 
		ReLU6(), 
		BatchNorm(), 1, "depthwise_m")
	depthwise_6 = SchemaNode(ShapeBound(None, None), 
		None,
		Sum(),
		Conv(2, 1, 6, 3, 1), 
		ReLU6(), 
		BatchNorm(), 1, "depthwise_l")
	shrink = SchemaNode(ShapeBound((32, 384), None), 
		None,
		Concat(), 
		Conv(1, 1, 1, 0, 1), 
		None, 
		BatchNorm(), 1, "shrink")
	down_sample_point = SchemaNode(ShapeBound((32, 384), (1, None)), 
		PowerGrowth(340, .5, .25),
		Sum(),
		Conv(1, 1), 
		SiLU(), 
		BatchNorm(), shrink_groups)
	down_sample_depthwise = SchemaNode(ShapeBound(None, (1, None)), 
		None,
		Sum(),
		Conv(2, 2, 1, 0, 1), 
		SiLU(), 
		BatchNorm())
	end = SchemaNode(ShapeBound(1, 1), 
		None,
		Sum(), 
		Full(), 
		None, 
		None)
	embed.add_group((second, 0, JoinType.NEW))

	second.add_group((skip, 3, JoinType.NEW), (expand_p, 0, JoinType.NEW), (expand_2, 0, JoinType.NEW))

	skip.add_group((skip, 3, JoinType.NEW), (expand_p, 0, JoinType.NEW), (expand_2, 0, JoinType.NEW), (expand_6, 0, JoinType.NEW))
	skip.add_group((skip, 3, JoinType.NEW), (expand_p, 0, JoinType.NEW), (expand_2, 0, JoinType.NEW))
	skip.add_group((down_sample_point, 0, JoinType.NEW))
	skip.add_group((end, 0, JoinType.NEW))
	expand_p.add_group((shrink, 3, JoinType.NEW))
	expand_2.add_group((depthwise_2, 1, JoinType.NEW))
	expand_6.add_group((depthwise_6, 1, JoinType.NEW))
	depthwise_2.add_group((shrink, 2, JoinType.EXISTING))
	depthwise_6.add_group((shrink, 2, JoinType.EXISTING))
	shrink.add_group((skip, 0, JoinType.EXISTING))

	down_sample_point.add_group((down_sample_depthwise, 0, JoinType.NEW))
	down_sample_depthwise.add_group((skip, 3, JoinType.NEW), (expand_p, 0, JoinType.NEW), (expand_2, 0, JoinType.NEW), (expand_6, 0, JoinType.NEW))
	down_sample_depthwise.add_group((down_sample_point, 0, JoinType.NEW))
	down_sample_depthwise.add_group((end, 0, JoinType.NEW))

	return Schema([embed], [end])





#ir = get_schema_c().compile_ir([LockedShape(CLASS_SIZE, TWEET_LENGTH)], BreedIndices(), ID(90))
#if ir is not None:
	print(generate_torch_module("M", ir))

#exit()
dataset = TweetDataset("data/twitter.csv")
SPLIT = 0.99
train, test = random_split(dataset, [int(len(dataset) * SPLIT), len(dataset) - int(len(dataset) * SPLIT)])
del dataset

control = Control(get_schema_b(), train, test, compile_models=False, max_id=ID(100),
	accuracy_function=lambda x, y: torch.sum((x > 0.5) == y).item() / len(y))
control.search([LockedShape(CLASS_SIZE, TWEET_LENGTH)], "./temp_saves", torch.nn.BCEWithLogitsLoss(),
	workers=4, batch_size=64, model_pool_size=5, training_epochs=18, breed_iterations=10, validation_multiple=1)
