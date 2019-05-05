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
def timeit_precsum(n_rows, n_cols, memory_layout, number=100):
   setup = precsum_setup.format(N=n_rows, M=n_cols, layout=memory_layout)
   # to level the field
   stmt = "res=np.empty({M}, dtype=np.float32); ps.pairwise_sum_2d(a,res,0)".format(M=n_cols)
   return min(timeit.repeat(stmt=stmt, setup=setup, repeat=5, number=number))/number



test_cases_L1 = ((2,1000,"C",100), (2,1000,"F",100),
                 (16,128,"C",100), (16,128,"F",100),
                 (50,50,"C",100), (50,50,"F",100),
                 (128,16,"C",100), (128,16,"F",100),
                 (1000,2,"C",100), (1000,2,"F",100),
                )

test_cases_L2 = ((2,4000,"C",100), (2,4000,"F",100),
                 (32,512,"C",100), (32,512,"F",100),
                 (120,120,"C",100), (120,120,"F",100),
                 (512,32,"C",100), (512,32,"F",100),
                 (4000,2,"C",100), (4000,2,"F",100),
                )

test_cases_L3 = ((2,64000,"C",100), (2,64000,"F",100),
                 (64,2000,"C",100), (64,2000,"F",100),
                 (400,400,"C",100), (400,400,"F",100),
                 (2000,64,"C",100), (2000,64,"F",100),
                 (64000,2,"C",100), (64000,2,"F",100),
                )

test_cases_MAIN = ((2,640000,"C",100), (2,640000,"F",100),
                   (64,20000,"C",100), (64,20000,"F",100),
                   (1000,1000,"C",100), (1000,1000,"F",100),
                   (20000,64,"C",100), (20000,64,"F",100),
                   (640000,2,"C",100), (640000,2,"F",100),
                )

for test_cases, name in zip([test_cases_L1, test_cases_L2, test_cases_L3, test_cases_MAIN], ["L1", "L2", "L3", "MAIN MEMORY"]):
    print("from",name,"cache:")
    for test_case in test_cases:
        numpy_time = timeit_numpy(*test_case)
        ps_time = timeit_precsum(*test_case)
        print(test_case,": np=", numpy_time, " ps=", ps_time, " factor numpy/ps=", numpy_time/ps_time)
