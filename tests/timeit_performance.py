import timeit
import numpy
import precsum as ps


#
numpy_setup = "import numpy as np; a=np.ones(({N},{M}), dtype=np.float32, order='{layout}');"
def timeit_numpy(n_rows, n_cols, memory_layout, number=100):
   setup = numpy_setup.format(N=n_rows, M=n_cols, layout=memory_layout)
   stmt = "a.sum(axis=0)"
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number


precsum_setup = "import precsum as ps;"+numpy_setup; 
def timeit_pairwise(n_rows, n_cols, memory_layout, number=100):
   setup = precsum_setup.format(N=n_rows, M=n_cols, layout=memory_layout)
   # to level the field
   stmt = "res=np.empty({M}, dtype=np.float32); ps.pairwise_sum_2d(a,res,0)".format(M=n_cols)
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

def timeit_kahan(n_rows, n_cols, memory_layout, number=100):
   setup = precsum_setup.format(N=n_rows, M=n_cols, layout=memory_layout)
   # to level the field
   stmt = "res=np.empty({M}, dtype=np.float32); ps.kahan_sum_2d(a,res,0)".format(M=n_cols)
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

def timeit_neumaier(n_rows, n_cols, memory_layout, number=100):
   setup = precsum_setup.format(N=n_rows, M=n_cols, layout=memory_layout)
   # to level the field
   stmt = "res=np.empty({M}, dtype=np.float32); ps.neumaier_sum_2d(a,res,0)".format(M=n_cols)
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number

NUMBER=20

test_cases_L1 = ((2,1000,"C",NUMBER), (2,1000,"F",NUMBER),
                 (16,128,"C",NUMBER), (16,128,"F",NUMBER),
                 (50,50,"C",NUMBER), (50,50,"F",NUMBER),
                 (128,16,"C",NUMBER), (128,16,"F",NUMBER),
                 (1000,2,"C",NUMBER), (1000,2,"F",NUMBER),
                )

test_cases_L2 = ((2,4000,"C",NUMBER), (2,4000,"F",NUMBER),
                 (32,512,"C",NUMBER), (32,512,"F",NUMBER),
                 (120,120,"C",NUMBER), (120,120,"F",NUMBER),
                 (512,32,"C",NUMBER), (512,32,"F",NUMBER),
                 (4000,2,"C",NUMBER), (4000,2,"F",NUMBER),
                )

test_cases_L3 = ((2,64000,"C",NUMBER), (2,64000,"F",NUMBER),
                 (64,2000,"C",NUMBER), (64,2000,"F",NUMBER),
                 (400,400,"C",NUMBER), (400,400,"F",NUMBER),
                 (2000,64,"C",NUMBER), (2000,64,"F",NUMBER),
                 (64000,2,"C",NUMBER), (64000,2,"F",NUMBER),
                )

test_cases_MAIN = ((2,640000,"C",NUMBER), (2,640000,"F",NUMBER),
                   (64,20000,"C",NUMBER), (64,20000,"F",NUMBER),
                   (1000,1000,"C",NUMBER), (1000,1000,"F",NUMBER),
                   (20000,64,"C",NUMBER), (20000,64,"F",NUMBER),
                   (640000,2,"C",NUMBER), (640000,2,"F",NUMBER),
                )

for test_cases, name in zip([test_cases_L1, test_cases_L2, test_cases_L3, test_cases_MAIN], ["L1", "L2", "L3", "MAIN MEMORY"]):
    print("from",name,"cache:")
    for test_case in test_cases:
        numpy_time = timeit_numpy(*test_case)
        pairwise_time = timeit_pairwise(*test_case)
        kahan_time = timeit_kahan(*test_case)
        neumaier_time = timeit_neumaier(*test_case)
        print(test_case,": ")
        print("   np=", numpy_time)
        print("   pairwise=", pairwise_time, " pairwise/numpy=", pairwise_time/numpy_time, " numpy/pairwise=", numpy_time/pairwise_time)
        print("   kahan=", kahan_time, " kahan/numpy=", kahan_time/numpy_time, " kahan/pairwise", kahan_time/pairwise_time)
        print("   neumaier=", neumaier_time, " neumaier/numpy=", neumaier_time/numpy_time, " neumaier/kahan", neumaier_time/kahan_time)




