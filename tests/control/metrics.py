import unittest

from src.control import Metrics, SampleCollection

class TestMetrics(unittest.TestCase):
	def setUp(self):
		pass
	def test_record(self):
		sample = SampleCollection(1.0, 1.0, 1.0, 0.0, 1)
		record = Metrics(2)
		record.record(sample)
		pass
	def test_reduce(self):
		pass
