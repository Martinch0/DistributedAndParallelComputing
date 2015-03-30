#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

template <typename T>
class CustomMatrix
{
	public:
		T** matrix;
		int height, width;

    CUDA_CALLABLE_MEMBER void initializeMatrix(int h, int w)
    {
			height = h;
			width = w;

      matrix = (T**) malloc (sizeof(T**) *height);
      for(int i=0; i<height; i++) {
        matrix[i] = (T*) malloc (sizeof(T*) * width);
      }
    }

    CUDA_CALLABLE_MEMBER void deleteMatrix()
    {
      for(int i=0; i<height; i++) {
        free(matrix[i]);
      }
      free(matrix);
    }

		CUDA_CALLABLE_MEMBER CustomMatrix(int h, int w)
		{
      initializeMatrix(h, w);
		}
		
		CUDA_CALLABLE_MEMBER ~CustomMatrix()
		{
      deleteMatrix();
		}
};

CUDA_CALLABLE_MEMBER long getIndex(long row, long col, long row_size)
{
  return row*row_size+col;
}

CUDA_CALLABLE_MEMBER long getSize(long row_size, long col_size)
{
  return row_size*col_size;
}
