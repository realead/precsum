import unittest

import precsum as ps
import numpy as np

import array
import ctypes


#special test cases for Kahan
class ḰahanSum1DTester(unittest.TestCase): 

   def test_sum(self):
      a=np.array([1,2,3], dtype=np.float32)
      s = ps.kahan_sum_1d(a)
      self.assertAlmostEqual(s, 6.0)

