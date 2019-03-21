#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


// random quick sort algorithm
__device__ int partition_GPU(unsigned char* window, int start, int end)
{
	int i = start;
	for(int j = start; j < end; j++)
		if(window[j] < window[end])
		{
			unsigned char temp = window[i]; window[i] = window[j]; window[j] = temp;
			i++;
		}
	unsigned char temp = window[i]; window[i] = window[end]; window[end] = temp;
	return i;
}


__device__ void randQuickSort(unsigned char * window, int start, int end, int idx)
{
	if(start < end)
	{
		// initialization of generating a random number.
		// cannot figure it out how to generate a random number in cuda
		// use the midpoint of start and end instead
/*	
		curandState state;	
		curand_init(1234, idx, 0, &state);
		// random number
		float myrandf = curand_uniform(&state);
		myrandf *= (end - start);
		int p = (int)myrandf;
*/
		int p = (end - start)/2 + start;
		unsigned char temp = window[p]; window[p] = window[end]; window[end] = temp;
		
		int q = partition_GPU(window, start, end);
		randQuickSort(window, start, q - 1, idx);
		randQuickSort(window, q + 1, end, idx);		
	}
}


/**
 * CUDA Kernel Device code
 */
__global__ void medianFilterGPU(const unsigned char *d_inputData, unsigned char *d_outputData, const int filterSize, const int img_height, const int img_width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int edgex = (filterSize / 2); 		
	int edgey = (filterSize / 2);
	if(y >= edgey && y < img_height - edgey && x >= edgex && x < img_width - edgex)	
	{
		int i = 0;
		unsigned char * window = new unsigned char[filterSize * filterSize];
		for(int fy = 0; fy < filterSize; fy++)
			for(int fx = 0; fx < filterSize; fx++)
			{
				window[i] = d_inputData[(y + fy - edgey) * img_width + (x + fx - edgex)];
				i++;
			}

		randQuickSort(window, 0, filterSize * filterSize - 1, x);
		d_outputData[y * img_width + x] = window[filterSize * filterSize / 2];	
		delete[] window;
	}
	else	// copy the pixels on the corners and sides
		d_outputData[y * img_width + x] = d_inputData[y * img_width + x];	
}











