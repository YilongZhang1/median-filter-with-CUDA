#include <cstdlib>				// rand
#include <algorithm>    		// std::swap
#include "medianFilter.h"

// declaration of random quick sort algorithm
int partition_CPU(unsigned char* window, int start, int end);
void randQuickSort(unsigned char * window, int start, int end);


unsigned char* medianFilterCPU(const unsigned char* h_inputData, const int img_height, const int img_width, const int filterSize)
{
	unsigned char * outputData_CPU = new unsigned char[img_height * img_width];
	unsigned char * window = new unsigned char[filterSize * filterSize];
	int edgex = (filterSize / 2); 
	int edgey = (filterSize / 2);
	for(int y = 0; y < img_height; y++)
		for(int x = 0; x < img_width; x++)
		{
			if(y >= edgey && y < img_height - edgey && x >= edgex && x < img_width - edgex)
			{
				int i = 0;
				for(int fy = 0; fy < filterSize; fy++)
				{
					for(int fx = 0; fx < filterSize; fx++)				
					{
						window[i] = h_inputData[(y + fy - edgey) * img_width + (x + fx - edgex)];
						i++;
					}				
				}
				randQuickSort(window, 0, filterSize * filterSize - 1);
				outputData_CPU[y * img_width + x] = window[filterSize * filterSize / 2];							
			}
			else
				outputData_CPU[y * img_width + x] = h_inputData[y * img_width + x];	
		}
	delete[] window;
	return outputData_CPU;
}


int partition_CPU(unsigned char* window, int start, int end)
{
	int i = start;
	for(int j = start; j < end; j++)
		if(window[j] < window[end])
		{
			std::swap(window[i], window[j]);
			i++;
		}
	std::swap(window[i], window[end]);
	return i;
}


void randQuickSort(unsigned char * window, int start, int end)
{
	if(start < end)
	{
		int p = std::rand() % (end - start + 1) + start;
		std::swap(window[p], window[end]);
		
		int q = partition_CPU(window, start, end);
		randQuickSort(window, start, q - 1);
		randQuickSort(window, q + 1, end);		
	}
}
