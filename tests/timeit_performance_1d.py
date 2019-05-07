import timeit
import numpy
import precsum as ps


#
common_setup = "import numpy as np; a=np.ones({N}, dtype=np.float32); import precsum as ps"

def timeit_numpy(n_rows, number=100):
   setup = common_setup.format(N=n_rows)
   stmt = "a.sum()"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

def timeit_pairwise(n_rows, number=100):
   setup = common_setup.format(N=n_rows)
   stmt = "ps.pairwise_sum_1d(a)"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

def timeit_kahan(n_rows, number=100):
   setup = common_setup.format(N=n_rows)
   stmt = "ps.kahan_sum_1d(a)"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

def timeit_neumaier(n_rows, number=100):
   setup = common_setup.format(N=n_rows)
   stmt = "ps.neumaier_sum_1d(a)"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number


def timeit_doubleprecision(n_rows, number=100):
   setup = common_setup.format(N=n_rows)
   stmt = "ps.doubleprecision_sum_1d(a)"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number


test_cases = ((10**1,100), (10**2, 100), (10**3, 100),
              (10**4,100), (10**5, 100), (10**6, 10),
              (10**7,10), #(10**8, 5),
             )



for test_case in test_cases:
    numpy_time = timeit_numpy(*test_case)
    pairwise_time = timeit_pairwise(*test_case)
    kahan_time = timeit_kahan(*test_case)
    neumaier_time = timeit_neumaier(*test_case)
    doubleprecision_time = timeit_doubleprecision(*test_case)
    print(test_case,":")
    print("  np=", numpy_time)
    print("  pairwise=", pairwise_time, " factor numpy/pairwise=", numpy_time/pairwise_time)
    print("  kahan=", kahan_time,  " factor pairwise/kahan=", pairwise_time/kahan_time)
    print("  neumaier=", neumaier_time,  " factor kahan/neumaier=", kahan_time/neumaier_time)
    print("  doubleprecision=", doubleprecision_time,  " factor pairwise/doubleprecision=", pairwise_time/doubleprecision_time)

