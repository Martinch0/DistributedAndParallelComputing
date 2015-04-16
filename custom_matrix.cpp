#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

CUDA_CALLABLE_MEMBER long getIndex(long row, long col, long row_size)
{
  return row*row_size+col;
}

CUDA_CALLABLE_MEMBER long getSize(long row_size, long col_size)
{
  return row_size*col_size;
}
