#include <vector>
using std::vector;

template <typename T>
class CustomMatrix
{
	public:
		vector<vector<T> > matrix;
		int height, width;

		CustomMatrix(int h, int w)
		{
			matrix.resize(h);
			for(int i=0; i < h; i++)
				matrix[i].resize(w);
			height = h;
			width = w;
		}
		
		~CustomMatrix()
		{
		}
};
