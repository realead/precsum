import unittest

import precsum as ps
import numpy as np
import array
import ctypes


class Sum2DTester(unittest.TestCase): 

   def test_sum_axis0(self):
      a=np.array([[1,2,3], [4,5,6]], dtype=np.float32)
      res=np.ones(3,dtype=np.float32)
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(res[0], 5.0)
      self.assertAlmostEqual(res[1], 7.0)
      self.assertAlmostEqual(res[2], 9.0)

   def test_sum_axis1(self):
      a=np.array([[1,2,3], [4,5,6]], dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      ps.pairwise_sum_2d(a,res,1)
      self.assertAlmostEqual(res[0], 6.0)
      self.assertAlmostEqual(res[1], 15.0)

   def sum_square(self, N, axis):  
      a=np.ones((N,N), dtype=np.float32)
      res=np.ones(N,dtype=np.float32)
      ps.pairwise_sum_2d(a,res,axis)
      self.assertAlmostEqual(res[N//2],float(N))

   def test_sum_axis0_9(self):
      self.sum_square(9,0)

   def test_sum_axis1_009(self):
      self.sum_square(9,1)

   def test_sum_axis0_015(self):
      self.sum_square(15,0)

   def test_sum_axis0_016(self):
      self.sum_square(16,0)

   def test_sum_axis1_016(self):
      self.sum_square(16,1)

   def test_sum_axis0_017(self):
      self.sum_square(17,0)

   def test_sum_axis0_020(self):
      self.sum_square(20,0)

   def test_sum_axis1_020(self):
      self.sum_square(20,1)

   def test_sum_axis0_127(self):
      self.sum_square(127,0)

   def test_sum_axis1_127(self):
      self.sum_square(127,1)

   def test_sum_axis0_128(self):
      self.sum_square(128,0)

   def test_sum_axis1_128(self):
      self.sum_square(128,1)

   def test_sum_axis0_129(self):
      self.sum_square(129,0)

   def test_sum_axis1_129(self):
      self.sum_square(129,1)

   def test_sum_axis0_1000(self):
      self.sum_square(1000,0)

   def test_sum_axis1_1000(self):
      self.sum_square(1000,1)


   def test_precise_sum_axis0(self):
      N=2*10**7
      a=np.ones((2,N), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      ps.pairwise_sum_2d(a,res,1)
      self.assertAlmostEqual(res[0], float(N))
      self.assertAlmostEqual(res[1], float(N))

   def test_precise_sum_axis1(self):
      N=2*10**7
      a=np.ones((N,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(res[0], float(N))
      self.assertAlmostEqual(res[1], float(N))


# errors:

   def test_non2d(self):
      a=np.ones((2,2,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("input must be a two-dimensional buffer", context.exception.args[0])

   def test_wrong_axis(self):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,2)
      self.assertEqual("unknown axis: 2", context.exception.args[0])

   def test_no1d_result(self):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones((2,2),dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("output must be an one-dimensional buffer", context.exception.args[0])

   def test_input_not_float32(self):
      a=np.ones((2,2), dtype=np.float64)
      res=np.ones(2, dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("input is not float32 data", context.exception.args[0])

   def test_output_not_float32(self):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(2, dtype=np.float64)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("output is not float32 data", context.exception.args[0])


   def test_dim_mismatch(self):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(3, dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("dimension missmatch between input(2) and output(3)", context.exception.args[0])


   def test_ouput_array(self):
      a=np.ones((2,2), dtype=np.float32)
      res=array.array('f', [4,3])
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(res[0], 2.0)
      self.assertAlmostEqual(res[1], 2.0)

   def test_ouput_ctypes(self):
      a=np.ones((2,2), dtype=np.float32)
      res=(ctypes.c_float * 2)()
      res[0]=1.0
      res[1]=1.0
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(res[0], 2.0)
      self.assertAlmostEqual(res[1], 2.0)


   def test_input_ctypes_ouput_array(self):
      a = ((ctypes.c_float * 2) * 2)()
      a[0][0]=1.0
      a[0][1]=2.0
      a[1][0]=3.0
      a[1][1]=4.0
      res=array.array('f', [4,3])
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(res[0], 4.0)
      self.assertAlmostEqual(res[1], 6.0)

#check all branches also for noncont

   def noncont_test(self, N):
      a=np.ones((N,N), dtype=np.float32)
      b=np.ones((N,2), dtype=np.float32)
      res=b[:,0]
      self.assertFalse(res.flags.contiguous)
      ps.pairwise_sum_2d(a,res,0)
      self.assertAlmostEqual(b[0,0], N)
      self.assertAlmostEqual(b[N//2,0], N)

   def test_ouput_noncontiguous_less8(self):
      self.noncont_test(5)

   def test_ouput_noncontiguous_8(self):
      self.noncont_test(8)

   def test_ouput_noncontiguous_mult8(self):
      self.noncont_test(64)

   def test_ouput_noncontiguous_mult8(self):
      self.noncont_test(70)

   def test_ouput_noncontiguous_128(self):
      self.noncont_test(128)

   def test_ouput_noncontiguous_larger128(self):
      self.noncont_test(1000)



