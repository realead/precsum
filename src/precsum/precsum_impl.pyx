
cimport cython
from cpython cimport buffer, PyBuffer_Release
# from libc.stdio cimport printf


### helper functions:

import sys

cdef class BufferHolder: 
    cdef buffer.Py_buffer view
    cdef bint buffer_set
    def __cinit__(self, obj, buffer_flags = buffer.PyBUF_FORMAT):
        buffer.PyObject_GetBuffer(obj, &(self.view), buffer_flags)
        buffer_set = 1

    def __dealloc__(self):
        if self.buffer_set:
            PyBuffer_Release(&(self.view)) 


# checks different possible formats for float32
cdef bint is_little_endian = (sys.byteorder == 'little')
cdef bint format_is_float32(const char *format):
    if format == NULL:
        return 0
    if format[0] == 0:
        return 0

    #the simple "f"-format:
    if format[0] == 102 and format[1]==0:
       return 1

    #the native "@f"-format:
    if format[0] == 64 and format[1] == 102 and format[2]==0:
       return 1

    if is_little_endian:
        #accepting little endian "<f"-format on my system:
        if format[0] == 60 and format[1] == 102 and format[2]==0:
            return 1
    else:
        #accepting big endian ">f"-format on my system:
        if format[0] == 62 and format[1] == 102 and format[2]==0:
            return 1

    return 0


####  1d summation:

# prototype:
# float pairwise_1dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride)
ctypedef float(*sum1d_type)(const float *, Py_ssize_t, Py_ssize_t)


@cython.cdivision(True) 
cdef sum_1d(object obj, sum1d_type worker): #returns PyObject, so the errors are propagated!
    #use buffer protocol:
    mem = BufferHolder(obj, buffer.PyBUF_FORMAT|buffer.PyBUF_STRIDES)
    if mem.view.ndim !=1:
        raise BufferError("can handle only one-dimensional buffers")
    if not format_is_float32(mem.view.format):
        raise BufferError("not float32 data")
    if mem.view.shape == NULL:
        raise BufferError("shape not set")
    if mem.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer")

    cdef Py_ssize_t stride = 1 if mem.view.strides == NULL  else mem.view.strides[0]/mem.view.itemsize
    return worker(<const float *>mem.view.buf, mem.view.shape[0], stride)


######### 2d summation:

# prototype: 
# void  pairwise_2dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride_along,  Py_ssize_t m, Py_ssize_t stride_crosswise, float *output, Py_ssize_t stride_output)
ctypedef void(*sum2d_type)(const float *, Py_ssize_t, Py_ssize_t,  Py_ssize_t, Py_ssize_t, float *, Py_ssize_t)

@cython.cdivision(True) 
cdef void sum_2d(object a, object output, int axis, sum1d_type worker1d, sum2d_type worker2d) except *:
    mem_input = BufferHolder(a,     buffer.PyBUF_FORMAT|buffer.PyBUF_STRIDES)
    mem_output = BufferHolder(output, buffer.PyBUF_FORMAT|buffer.PyBUF_STRIDES)
    if mem_input.view.ndim !=2:
        raise BufferError("input must be a two-dimensional buffer") 
    if mem_input.view.ndim <= axis:
        raise BufferError("unknown axis: {0}".format(axis))    
    if mem_output.view.ndim !=1:
        raise BufferError("output must be an one-dimensional buffer")
    if not format_is_float32(mem_input.view.format):
        raise BufferError("input is not float32 data")
    if not format_is_float32(mem_output.view.format):
        raise BufferError("output is not float32 data")
    if mem_input.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer as input")
    if mem_output.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer as ouput")


    cdef Py_ssize_t N = mem_input.view.shape[0] if axis == 0  else mem_input.view.shape[1]
    cdef Py_ssize_t M = mem_input.view.shape[1] if axis == 0  else mem_input.view.shape[0]

    if M != mem_output.view.shape[0]:
        raise BufferError("dimension missmatch between input({0}) and output({1})".format(M, mem_output.view.shape[0]))

    
    ## handle strides:
    cdef Py_ssize_t stride_N, stride_M
    cdef Py_ssize_t stride_N_in_bytes, stride_M_in_bytes
    # strides = NULL implies the usual C-memory layout:
    # is for example used by ctypes
    if mem_input.view.strides == NULL:
        stride_N = mem_input.view.shape[1] if axis == 0  else 1
        stride_M = 1 if axis == 0  else mem_input.view.shape[1]
    else:
        stride_N_in_bytes = mem_input.view.strides[0] if axis == 0  else mem_input.view.strides[1]
        stride_M_in_bytes = mem_input.view.strides[1] if axis == 0  else mem_input.view.strides[0]
        stride_N = stride_N_in_bytes/mem_input.view.itemsize
        stride_M = stride_M_in_bytes/mem_input.view.itemsize

    cdef Py_ssize_t stride_output = 1 #  default value, if strides unset
    if mem_output.view.strides != NULL:
        stride_output = mem_output.view.strides[0]/mem_output.view.itemsize

    cdef const float * input_buf = <const float *>mem_input.view.buf
    cdef float * output_buf = <float *>mem_output.view.buf
    cdef Py_ssize_t i
    if stride_N < stride_M:
        for i in range(M):
            output_buf[i*stride_output] = worker1d(input_buf, N, stride_N)
            input_buf+=stride_M
        
    else:       
        worker2d(input_buf, N, stride_N, M, stride_M, output_buf, stride_output)


################# pairwise summation:

cdef extern from "pairwise_sum.c":
    float pairwise_1dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride)
    void  pairwise_2dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride_along,  Py_ssize_t m, Py_ssize_t stride_crosswise, float *output, Py_ssize_t stride_output)


################# kahan summation:

cdef extern from "kahan_sum.c":
    float kahan_1dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride)
    void  kahan_2dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride_along,  Py_ssize_t m, Py_ssize_t stride_crosswise, float *output, Py_ssize_t stride_output)

################# kahan summation:

cdef extern from "neumaier_sum.c":
    float neumaier_1dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride)
    #void  kahan_2dsum_FLOAT(const float *ptr, Py_ssize_t n, Py_ssize_t stride_along,  Py_ssize_t m, Py_ssize_t stride_crosswise, float *output, Py_ssize_t stride_output)



############  python interface:

def pairwise_sum_1d(object obj):
    return sum_1d(obj, pairwise_1dsum_FLOAT)

def kahan_sum_1d(object obj):
    return sum_1d(obj, kahan_1dsum_FLOAT)

def neumaier_sum_1d(object obj):
    return sum_1d(obj, neumaier_1dsum_FLOAT)


def pairwise_sum_2d(object a, object output, int axis):
    sum_2d(a, output, axis, pairwise_1dsum_FLOAT, pairwise_2dsum_FLOAT)

def kahan_sum_2d(object a, object output, int axis):
    sum_2d(a, output, axis, kahan_1dsum_FLOAT, kahan_2dsum_FLOAT)






