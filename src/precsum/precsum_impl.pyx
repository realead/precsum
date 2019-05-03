
from cpython cimport buffer, PyBuffer_Release


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
    if mem_input.view.format == NULL or mem_input.view.format[0]!=102 or mem_input.view.format[1]!=0:
        raise BufferError("input is not float32 data")
    if mem_output.view.format == NULL or mem_output.view.format[0]!=102 or mem_output.view.format[1]!=0:
        raise BufferError("output is not float32 data")
    if mem_input.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer as input")
    if mem_output.view.suboffsets != NULL:
        raise BufferError("cannot handle indirect buffer as ouput")

    cdef unsigned int N = mem_input.view.shape[0] if axis == 0  else mem_input.view.shape[1]
    cdef unsigned int M = mem_input.view.shape[1] if axis == 0  else mem_input.view.shape[0]
    cdef unsigned int stride_N = mem_input.view.strides[0] if axis == 0  else mem_input.view.strides[1]
    cdef unsigned int stride_M = mem_input.view.strides[1] if axis == 0  else mem_input.view.strides[0]

    if M != mem_output.view.shape[0]:
        raise BufferError("dimension missmatch between input({0}) and output({1})".format(M, mem_output.view.shape[0]))

    cdef unsigned int item_size = mem_input.view.itemsize

    pairwise_2dsum_FLOAT(<const float *>mem_input.view.buf, <float *>mem_output.view.buf, N, stride_N//item_size, M, stride_M//item_size)





