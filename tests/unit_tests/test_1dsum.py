import unittest

import precsum as ps
import numpy as np

import array
import ctypes

import uttemplate

@uttemplate.from_templates([ps.pairwise_sum_1d, ps.kahan_sum_1d])
class Sum1DTester(unittest.TestCase): 

   def template_sum(self, fun):
      a=np.array([1,2,3], dtype=np.float32)
      s = fun(a)
      self.assertAlmostEqual(s, 6.0)

   def template_prec_sum(self, fun):
      a=np.ones(10**8, dtype=np.float32)
      s = fun(a)
      self.assertAlmostEqual(s, 1.0*10**8)

   def template_bigger_stride(self, fun):
     a=np.ones((5,5), dtype=np.float32)
     a[:,0]=3.0;
     self.assertAlmostEqual(fun(a[:,0]), 15.0)     
     a[0,:]=2.0
     self.assertAlmostEqual(fun(a[0,:]), 10.0)

   def template_sum_int(self, fun):
      a=np.ones([1], dtype=np.int32)
      with self.assertRaises(BufferError) as context:
            fun(a)
      self.assertEqual("not float32 data", context.exception.args[0])

   def template_sum_2d(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            fun(a)
      self.assertEqual("can handle only one-dimensional buffers", context.exception.args[0])

   def template_array(self, fun):
      a=array.array('f', [4,3])
      res=fun(a)
      self.assertAlmostEqual(res, 7.0)


   def template_ctypes(self, fun):
      a = (ctypes.c_float * 2)()
      a[0]=1.0
      a[1]=2.0
      res=fun(a)
      self.assertAlmostEqual(res, 3.0)


########  different sizes:

   def diff_sizes_test(self, fun, N):
      a=np.ones(N, dtype=np.float32)
      self.assertAlmostEqual(fun(a), N)


   def template_less8(self, fun):
      self.diff_sizes_test(fun, 7)

   def template_mulitple8(self, fun):
      self.diff_sizes_test(fun, 64)

   def template_non_multiple8(self, fun):
      self.diff_sizes_test(fun, 127)

   def template_128(self, fun):
      self.diff_sizes_test(fun, 128)

   def template_mulitple128(self, fun):
      self.diff_sizes_test(fun, 512)

   def template_non_mulitple128(self, fun):
      self.diff_sizes_test(fun, 1000)

   def template_non_multiple8_bruteforce(self, fun):
      for i in range(1,500):
          self.diff_sizes_test(fun, i)

######## input non-contiguous:

   def diff_sizes_noncont_test(self, fun, N):
      a=np.ones((N,3), dtype=np.float32)
      a[:,1]=2.0
      a[:,2]=3.0
      b=a[:,0]
      self.assertFalse(b.flags.contiguous)
      self.assertAlmostEqual(fun(b), N)


   def template_less8_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 7)

   def template_mulitple8_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 64)

   def template_non_multiple8_noncont_bruteforce(self, fun):
      for i in range(2,500):
          self.diff_sizes_noncont_test(fun, i)

   def template_non_multiple8_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 127)

   def template_128_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 128)

   def template_mulitple128_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 512)

   def template_non_mulitple128_noncont(self, fun):
      self.diff_sizes_noncont_test(fun, 1000)
