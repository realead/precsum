
from cpython cimport buffer, PyBuffer_Release


cdef extern from "sum1d.c":
    float pairwise_1dsum_FLOAT(const float *ptr, unsigned int n, unsigned int stride)



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
