# precsum

precise summation for floating-point arrays

## About:

   A collection of fast and precise summation methods for 1D- and 2D-arrays. Motivated by this SO-post: https://stackoverflow.com/q/55512278/5769463.

   Supports object with buffer-interface consisting of floats (and maybe doubles in the future).


## Dependencies:

Essentials: 

  - Python 3 (tested with Python 3.7)
  - setuptools
  - Cython
  - c-build chain

Additional dependencies for tests:
   
  - `sh`
  - `virtualenv`
  - numpy + some other python libraries.

## Instalation:

To install the module using pip run:

    pip install https://github.com/realead/precsum/zipball/master

It is possible to uninstall it afterwards via

    pip uninstall precsum

You can also install using the setup.py file from the root directory of the project:

    python setup.py install

However, there is no easy way to deinstall it afterwards (only manually) if setup.py was used directly.

## Usage

### 1D sum

For any object with support for buffer protocol, here for example a numpy array:

    >>> import precsum as ps
    >>> import numpy as np
    >>> a=np.ones(100, dtype=np.float32)
    >>> ps.pairwise_sum_1d(a)
    100

There are following methods supporting summation of float32-arrays:

 * `pairwise_sum_1d` : kind of pairwise summation, the same algorithmus as used by numpy
 * `kahan_sum_1d` :  kahan-summation algorithm
 * `neumaier_sum_1d`: neumaier-summation algorithm
 * `doubleprecision_sum_1d` accumulator uses double-precision


### 2D sum along an axis

Uses input and ouput buffers, for example numpy-arrays:

    >>> import precsum as ps
    >>> import numpy as np
    >>> a=np.ones(3,2, dtype=np.float32)
    >>> out=np.empty(2, dtype=np.float32)
    >>> ps.pairwise_sum_2d(a,out,axis=0)
    >>> out
    [3.0, 3.0]

There are following methods supporting summation of float32-arrays:

 * `pairwise_sum_2d` : kind of pairwise summation, the same algorith as used by numpy in 1D-case
 * `kahan_sum_1d` :  kahan-summation algorithm
 * `neumaier_sum_1d`: neumaier-summation algorithm

which all have signature:

    fun(input_2d_array, output_1d_array, axis)

`input_2d_array` and `output_2d_array` must support buffer protocol and must not be indirect.


## Performance

### Precision 

#### Theoretical precision

Given condition number  C of the series `x1..xn` as `(|x1|+...+|xn|)/|x1+...+xn|`, the errors are known to be:

    Kind of summation        |   worst-case      | average-case (random walk) |
    -------------------------------------------------------------------------
    naive summation          |   O(n*eps*C)      |   O(sqrt(n)*eps*C)
    pairwise summation       |   O(log(n)*eps*C) |   O(sqrt(log(n))*eps*C)
    Kahan summation          |   O(eps*C)        |
    Neumaier summation       |   O(eps*C)        |
    Double-prec accumulator  |   O(n*eps^2*C)    |   O(sqrt(n)*eps^2*C)


#### Experimental precision (float)

The experiments (series with  10^7 elements) is in accordance with theoretical predictions:


    positive series:
    name	average rel. error	max rel. error:
       numpy.sum()	 4.384003211603486e-12 	 1.3661819465920785e-06
       pairwise()	 4.099447461251432e-13 	 1.0762488961755901e-07
       kahan()	 2.705843198719916e-13 	 6.727859740547816e-08
       neumaier()	 2.705843198719916e-13 	 6.727859740547816e-08
       doubleprec()	 2.5402501621155227e-13 	 4.940517637843998e-08
    alternating series:
    name	average error	max error:
       numpy.sum()	 4.361395579960003e-09 	 0.0021215926308286726
       pairwise()	 9.806826391212554e-10 	 0.00030505847837503097
       kahan()	 3.359827809020999e-10 	 0.00013471003012455185
       neumaier()	 2.6671181185804473e-10 	 9.460922478865541e-05
       doubleprec()	 1.5377834542675828e-10 	 8.82034728419967e-05

Somewhat unexpected is, that numpy's sum has a bigger error than pairwise, which is basically a rip-off of the numpy's implementation! Neumaier performs better than kahan for alternating series.


### Running times

#### 1D, float

Noteworthy details (see `tests/timeit_performance_1d.py`):

  * For short series `pairwise` beats numpy's version easily due to smaller overhead
  * For long series (10^7), numpy's version is 10% faster than `pairwise`(don't know yet why - different compiler, different optimizations?)
  * `doubleprec` is about 10-20% slower than `pairwise`
  * `kahan` is 4 times slower than `pairwise`
  * `neumaier` is 20-30% slower than `kahan`


#### 2D, float

Noteworthy details (see `tests/timeit_performance.py`):

  * `np.sum` uses pairwise summation only when summing along contignuous axis, `precsum`-methods always.
  * the relation between `precsum`-performances is similar to 1D case.
  * the relation between `pairwise` and `np.sum` is similar to 1D case when summing along contignuous axis.
  * when summing along non-contignuous axis, `pairwise` is worse for short series (about 1000 elements, up to factor 3 slower), but much faster for long series  (about 100000 elements, up to factor 10).

The reason for that is, that `pairwise` has a less efficient code, which uses better the caching:

  * numpy adds rowwise, which means less read-cache-misses, but the resulting vector gets evicted from the cache once the dimension becomes long.
  * `pairwise`loads memory in tiles and the accumulator stays hot in cache even for larger series.

    
## Testing:

For testing of the local version run:

    sh test_install.sh

in the `tests` subfolder.

For testing of the version from github run:

    sh test_install.sh from-github

For keeping the the virtual enviroment after the tests:

    sh test_install.sh local keep

## Versions:

  0.1.0: 1D an 2D summations for float32

## Outlook:

  * for float32
  * for dimensions > 2
