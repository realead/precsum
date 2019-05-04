import unittest

import precsum as ps
import numpy as np

import array
import ctypes


class Sum1DTester(unittest.TestCase): 

   def test_sum(self):
      a=np.array([1,2,3], dtype=np.float32)
      s = ps.pairwise_sum_1d(a)
      self.assertAlmostEqual(s, 6.0)

   def test_prec_sum(self):
      a=np.ones(10**8, dtype=np.float32)
      s = ps.pairwise_sum_1d(a)
      self.assertAlmostEqual(s, 1.0*10**8)

   def test_bigger_stride(self):
     a=np.ones((5,5), dtype=np.float32)
     a[:,0]=3.0;
     self.assertAlmostEqual(ps.pairwise_sum_1d(a[:,0]), 15.0)     
     a[0,:]=2.0
     self.assertAlmostEqual(ps.pairwise_sum_1d(a[0,:]), 10.0)

   def test_sum_int(self):
      a=np.ones([1], dtype=np.int32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_1d(a)
      self.assertEqual("not float32 data", context.exception.args[0])

   def test_sum_2d(self):
      a=np.ones((2,2), dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_1d(a)
      self.assertEqual("can handle only one-dimensional buffers", context.exception.args[0])

   def test_array(self):
      a=array.array('f', [4,3])
      res=ps.pairwise_sum_1d(a)
      self.assertAlmostEqual(res, 7.0)


   def test_ctypes(self):
      a = (ctypes.c_float * 2)()
      a[0]=1.0
      a[1]=2.0
      res=ps.pairwise_sum_1d(a)
      self.assertAlmostEqual(res, 3.0)

