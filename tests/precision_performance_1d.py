import numpy as np
import precsum as ps

def printout(results, label):
    average_errors = results.mean(axis=1)
    max_errors =  results.max(axis=1)

    print("name\taverage "+label+"error\tmax "+label+"error:")
    print("   numpy.sum()\t",average_errors[0],"\t",max_errors[0])
    print("   pairwise()\t",average_errors[1],"\t",max_errors[1])
    print("   kahan()\t",average_errors[2],"\t",max_errors[2])
    print("   neumaier()\t",average_errors[3],"\t",max_errors[3])
    print("   doubleprec()\t",average_errors[4],"\t",max_errors[4])
   
def positive():
    N=100
    SIZE=10**7
    np.random.seed(42)


    results=np.empty((5,SIZE), dtype=np.float64)
    for i in range(N):
       values = np.random.rand(SIZE).astype(np.float32)
       golden = np.sum(values.astype(np.float64))
       results[0,i]=abs(np.sum(values)-golden)/abs(golden)
       results[1,i]=abs(ps.pairwise_sum_1d(values)-golden)/abs(golden)
       results[2,i]=abs(ps.kahan_sum_1d(values)-golden)/abs(golden)
       results[3,i]=abs(ps.neumaier_sum_1d(values)-golden)/abs(golden)
       results[4,i]=abs(ps.doubleprecision_sum_1d(values)-golden)/abs(golden)


    print("positive series:")
    printout(results, "rel. ")


def alternating():
    N=100
    SIZE=10**7
    np.random.seed(42)


    results=np.empty((5,SIZE), dtype=np.float64)
    for i in range(N):
       values = (np.random.rand(SIZE)-.5).astype(np.float32)
       #will be about 0.0, we are interested in absolute error (otherwise / almost 0.0 makes them less meaningful)
       golden = np.sum(values.astype(np.float64))    
       results[0,i]=abs(np.sum(values)-golden)
       results[1,i]=abs(ps.pairwise_sum_1d(values)-golden)
       results[2,i]=abs(ps.kahan_sum_1d(values)-golden)
       results[3,i]=abs(ps.neumaier_sum_1d(values)-golden)
       results[4,i]=abs(ps.doubleprecision_sum_1d(values)-golden)


    print("alternating series:")
    printout(results,"")

positive()
alternating()

