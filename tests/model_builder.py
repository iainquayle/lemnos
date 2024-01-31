import unittest

from src.model.model import ModelNode, Model 
from src.model.model_builder import ModelBuilder, _ExpansionCollection, _ExpansionNode, _ExpansionStack
from src.shared.shape import Bound, LockedShape 
from src.schema.schema_node import SchemaNode, Transition, TransitionGroup
from src.shared.index import Index

class TestNodeBuild(unittest.TestCase):
	def setUp(self) -> None:
		pass
	def test_empty_expand(self):
		self.assertTrue(not ModelBuilder._build_node(_ExpansionCollection(), [Index()], 0))
	def test_single_expand(self):
		collection = _ExpansionCollection()
		#collection.add
		pass
	def test_fail_single_expand(self):
		pass

