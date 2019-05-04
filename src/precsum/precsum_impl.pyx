
from cpython cimport buffer, PyBuffer_Release

import sys

cdef extern from "sum1d.c":
    float pairwise_1dsum_FLOAT(const float *ptr, unsigned int n, unsigned int stride)
    void  pairwise_2dsum_FLOAT(const float *ptr, float *output, unsigned int n, unsigned int stride_along, unsigned int m, unsigned int stride_crosswise)



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

    if sys.byteorder == 'little':
        #accepting little endian "<f"-format on my system:
        if format[0] == 60 and format[1] == 102 and format[2]==0:
            return 1
    else:
        #accepting big endian ">f"-format on my system:
        if format[0] == 62 and format[1] == 102 and format[2]==0:
            return 1

    return 0


def pairwise_sum_1d(object obj):
    #use buffer protocol:
    mem = BufferHolder(obj, buffer.PyBUF_FORMAT|buffer.PyBUF_STRIDES)
    if mem.view.ndim !=1:
        raise BufferError("can handle only one-dimensional buffers")
    if mem.view.format == NULL or mem.view.format[0]!=102 or mem.view.format[1]!=0:
        raise BufferError("not float32 data")
    if mem.view.shape == NULL:
        raise BufferError("shape not set")
    if mem.view.strides == NULL:
        raise BufferError("stride not set")
    if mem.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer")

    return pairwise_1dsum_FLOAT(<const float *>mem.view.buf, mem.view.shape[0], mem.view.strides[0]//mem.view.itemsize)




def pairwise_sum_2d(object a, object output, unsigned int axis):
    #use buffer protocol:
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


    cdef unsigned int N = mem_input.view.shape[0] if axis == 0  else mem_input.view.shape[1]
    cdef unsigned int M = mem_input.view.shape[1] if axis == 0  else mem_input.view.shape[0]

    if M != mem_output.view.shape[0]:
        raise BufferError("dimension missmatch between input({0}) and output({1})".format(M, mem_output.view.shape[0]))

    
    ## handle strides:
    cdef unsigned int stride_N, stride_M
    cdef unsigned int stride_N_in_bytes, stride_M_in_bytes
    # strides = NULL implies the usual C-memory layout:
    # is for example used by ctypes
    if mem_input.view.strides == NULL:
        stride_N = mem_input.view.shape[1] if axis == 0  else 1
        stride_M = 1 if axis == 0  else mem_input.view.shape[1]
    else:
        stride_N_in_bytes = mem_input.view.strides[0] if axis == 0  else mem_input.view.strides[1]
        stride_M_in_bytes = mem_input.view.strides[1] if axis == 0  else mem_input.view.strides[0]
        stride_N = stride_N_in_bytes//mem_input.view.itemsize
        stride_M = stride_M_in_bytes//mem_input.view.itemsize


    pairwise_2dsum_FLOAT(<const float *>mem_input.view.buf, <float *>mem_output.view.buf, N, stride_N, M, stride_M)





