import unittest

import precsum as ps
import numpy as np
import array
import ctypes

import uttemplate

@uttemplate.from_templates([ps.pairwise_sum_2d, ps.kahan_sum_2d])
class Sum2DTester(unittest.TestCase): 

   def template_sum_axis0(self, fun):
      a=np.array([[1,2,3], [4,5,6]], dtype=np.float32)
      res=np.ones(3,dtype=np.float32)
      fun(a,res,0)
      self.assertAlmostEqual(res[0], 5.0)
      self.assertAlmostEqual(res[1], 7.0)
      self.assertAlmostEqual(res[2], 9.0)

   def template_sum_axis1(self, fun):
      a=np.array([[1,2,3], [4,5,6]], dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      fun(a,res,1)
      self.assertAlmostEqual(res[0], 6.0)
      self.assertAlmostEqual(res[1], 15.0)

   def sum_square(self, N, axis,fun):  
      a=np.ones((N,N), dtype=np.float32)
      res=np.ones(N,dtype=np.float32)
      fun(a,res,axis)
      self.assertAlmostEqual(res[N//2],float(N))

   def template_sum_axis0_9(self, fun):
      self.sum_square(9,0,fun)

   def template_sum_axis1_009(self, fun):
      self.sum_square(9,1,fun)

   def template_sum_axis0_015(self, fun):
      self.sum_square(15,0,fun)

   def template_sum_axis0_016(self, fun):
      self.sum_square(16,0,fun)

   def template_sum_axis1_016(self, fun):
      self.sum_square(16,1,fun)

   def template_sum_axis0_017(self, fun):
      self.sum_square(17,0,fun)

   def template_sum_axis0_020(self, fun):
      self.sum_square(20,0,fun)

   def template_sum_axis1_020(self, fun):
      self.sum_square(20,1,fun)

   def template_sum_axis0_127(self, fun):
      self.sum_square(127,0,fun)

   def template_sum_axis1_127(self, fun):
      self.sum_square(127,1,fun)

   def template_sum_axis0_128(self, fun):
      self.sum_square(128,0,fun)

   def template_sum_axis1_128(self, fun):
      self.sum_square(128,1,fun)

   def template_sum_axis0_129(self, fun):
      self.sum_square(129,0,fun)

   def template_sum_axis1_129(self, fun):
      self.sum_square(129,1,fun)

   def template_sum_axis0_1000(self, fun):
      self.sum_square(1000,0,fun)

   def template_sum_axis1_1000(self, fun):
      self.sum_square(1000,1,fun)


   def template_precise_sum_axis0(self, fun):
      N=2*10**7
      a=np.ones((2,N), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      fun(a,res,1)
      self.assertAlmostEqual(res[0], float(N))
      self.assertAlmostEqual(res[1], float(N))

   def template_precise_sum_axis1(self, fun):
      N=2*10**7
      a=np.ones((N,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      fun(a,res,0)
      self.assertAlmostEqual(res[0], float(N))
      self.assertAlmostEqual(res[1], float(N))


# errors:

   def template_non2d(self, fun):
      a=np.ones((2,2,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            fun(a,res,0)
      self.assertEqual("input must be a two-dimensional buffer", context.exception.args[0])

   def template_wrong_axis(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(2,dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            fun(a,res,2)
      self.assertEqual("unknown axis: 2", context.exception.args[0])

   def template_no1d_result(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones((2,2),dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            fun(a,res,0)
      self.assertEqual("output must be an one-dimensional buffer", context.exception.args[0])

   def template_input_not_float32(self, fun):
      a=np.ones((2,2), dtype=np.float64)
      res=np.ones(2, dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            ps.pairwise_sum_2d(a,res,0)
      self.assertEqual("input is not float32 data", context.exception.args[0])

   def template_output_not_float32(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(2, dtype=np.float64)
      with self.assertRaises(BufferError) as context:
            fun(a,res,0)
      self.assertEqual("output is not float32 data", context.exception.args[0])


   def template_dim_mismatch(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=np.ones(3, dtype=np.float32)
      with self.assertRaises(BufferError) as context:
            fun(a,res,0)
      self.assertEqual("dimension missmatch between input(2) and output(3)", context.exception.args[0])


   def template_ouput_array(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=array.array('f', [4,3])
      fun(a,res,0)
      self.assertAlmostEqual(res[0], 2.0)
      self.assertAlmostEqual(res[1], 2.0)

   def template_ouput_ctypes(self, fun):
      a=np.ones((2,2), dtype=np.float32)
      res=(ctypes.c_float * 2)()
      res[0]=1.0
      res[1]=1.0
      fun(a,res,0)
      self.assertAlmostEqual(res[0], 2.0)
      self.assertAlmostEqual(res[1], 2.0)


   def template_input_ctypes_ouput_array(self, fun):
      a = ((ctypes.c_float * 2) * 2)()
      a[0][0]=1.0
      a[0][1]=2.0
      a[1][0]=3.0
      a[1][1]=4.0
      res=array.array('f', [4,3])
      fun(a,res,0)
      self.assertAlmostEqual(res[0], 4.0)
      self.assertAlmostEqual(res[1], 6.0)

#check all branches also for noncont

   def noncont_test(self, N, fun):
      a=np.ones((N,N), dtype=np.float32)
      b=np.ones((N,2), dtype=np.float32)
      res=b[:,0]
      self.assertFalse(res.flags.contiguous)
      fun(a,res,0)
      self.assertAlmostEqual(b[0,0], N)
      self.assertAlmostEqual(b[N//2,0], N)

   def template_ouput_noncontiguous_less8(self, fun):
      self.noncont_test(5, fun)

   def template_ouput_noncontiguous_8(self, fun):
      self.noncont_test(8, fun)

   def template_ouput_noncontiguous_mult8(self, fun):
      self.noncont_test(64, fun)

   def template_ouput_noncontiguous_mult8(self, fun):
      self.noncont_test(70, fun)

   def template_ouput_noncontiguous_128(self, fun):
      self.noncont_test(128, fun)

   def template_ouput_noncontiguous_larger128(self, fun):
      self.noncont_test(1000, fun)



####### random-tests:

   def random_test(self, fun, seed):
       np.random.seed(seed)
       N,M=np.random.randint(3,1000,size=2)
       A=np.random.randint(3,1000,size=(N,M)).astype(np.float32)
       
       for axis in [0,1]:
           Nres = A.shape[1] if axis == 0 else A.shape[0]
           res_np=A.sum(axis=axis)     
           res_ps=np.empty(Nres, dtype=np.float32)
           fun(A, res_ps, axis)
           np.testing.assert_almost_equal(res_np, res_ps)



   def template_random_tests(self, fun):
        for i in range(200):
            self.random_test(fun, i)


