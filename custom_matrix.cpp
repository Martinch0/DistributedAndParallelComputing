#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

// Returns the corresponding index of a 2D matrix point into a 1D array
CUDA_CALLABLE_MEMBER long getIndex(long row, long col, long row_size)
{
  return row*row_size+col;
}

// Returns the number of elements of a 2D matrix
CUDA_CALLABLE_MEMBER long getSize(long row_size, long col_size)
{
  return row_size*col_size;
}
