import unittest

from lemnos.control import Metrics, ResultsSample

class TestMetrics(unittest.TestCase):
	def setUp(self):
		self.sample_1 = ResultsSample(1.0, 1.0, 1.0, 0, None, None, 1)
		self.sample_2 = ResultsSample(2.0, 2.0, 2.0, 1, None, None, 1)
		pass
	def test_init(self):
		sample = ResultsSample(1.0, 1.0, 1.0, 0, None, None, 1)
		sample_2 = ResultsSample(2.0, 2.0, 2.0, 1, None, None, 2)
		self.assertEqual(sample.total_loss, 1.0)
		self.assertEqual(sample_2.total_loss, 4.0)
	def test_merge(self):
		merged = self.sample_1.merge(self.sample_2)
		self.assertEqual(merged.total_loss, 3.0)
		self.assertEqual(merged.max_loss, 2.0)
		self.assertEqual(merged.min_loss, 1.0)
		self.assertEqual(merged.correct, 1)
		self.assertEqual(merged.sample_size, 2)
		self.assertEqual(merged.time, None)
		self.assertEqual(merged.epoch, None)
	def test_record(self):
		record = Metrics(2)
		record.record(self.sample_1)
		record.record(self.sample_2)
		self.assertEqual(record._samples[0].total_loss, 1.0)
		record.record(self.sample_1)
		self.assertEqual(len(record._samples), 2)
		self.assertEqual(record._samples[-1].total_loss, 1.0)
		record.record(self.sample_2)
		self.assertEqual(record._samples[-1].total_loss, 3.0)
	def test_reduce(self):
		record = Metrics(4)
		for _ in range(4):
			record.record(self.sample_1)
		self.assertEqual(len(record._samples), 4)
		record.record(self.sample_1)
		self.assertEqual(len(record._samples), 3)
		record.record(self.sample_1)
		self.assertEqual(len(record._samples), 3)
		self.assertEqual(record._samples[-1].sample_size, 2)
