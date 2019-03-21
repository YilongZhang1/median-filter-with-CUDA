#include <vector>
#include <boost/algorithm/string.hpp>	// boost::split, boost::is_any_of
#include <chrono>	// timing

#include <helper_functions.h> // includes for SDK helper functions
#include <helper_image.h>		// sdkLoadPGM, sdkLoadPPM4, and sdkSavePGM
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include <cuda_runtime.h>

#include "medianFilter.h"
#include "device.cuh"

#define timing 0

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		std::cout << "usage: ./median_filter_CUDA <filter size> <input filename> <output filename>" << std::endl;
		return 1;
	}
	
	// parse input parameters
	int filterSize = atoi(argv[1]);
	char * inputFileName = argv[2];
	char * outputFileName = argv[3];
	unsigned int img_height, img_width;
	// has to be unsigned char type to be consistent with sdkLoad functions
	unsigned char * h_inputData = new unsigned char[IMG_SIZE * IMG_SIZE];
	
	bool successLoad;
	
	// parse and check the input filename
	std::vector<std::string> nameParts;
	boost::split(nameParts, inputFileName, boost::is_any_of("."));
	if(nameParts.back() == "pgm")
		successLoad = sdkLoadPGM<unsigned char>(inputFileName, &h_inputData, &img_width, &img_height);
	else if(nameParts.back() == "ppm")
		successLoad = sdkLoadPPM4<unsigned char>(inputFileName, &h_inputData, &img_width, &img_height);
	else
	{
		std::cout<<"Image loading failure. Please check input image extension"<<std::endl;
		return 1;		
	}
	
	if(successLoad != true)
	{
		std::cout<<"Image loading failure."<<std::endl;
		return 1;
	}
	std::cout << "\nSucceed in image loading. Image information: "<< std::endl;
	std::cout<<"\t image name: "<<inputFileName<<std::endl;
	std::cout<<"\t image size: "<<img_width <<" * "<<img_height<<std::endl;
	
	// realize the median filter in CPU
	// deep copy inputData into outputData
	unsigned char * outputData_CPU = medianFilterCPU(h_inputData, img_height, img_width, filterSize);

	// realize the median filter in GPU
	// deep copy inputData into outputData
	unsigned char * outputData_GPU = new unsigned char[img_height * img_width];

	// Allocate the device inputData
	cudaError_t err = cudaSuccess;
    unsigned char *d_inputData = NULL;
    err = cudaMalloc((void **)&d_inputData, sizeof(unsigned char)*img_height*img_width);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Allocate the device outputData
    unsigned char *d_outputData = NULL;
    err = cudaMalloc((void **)&d_outputData, sizeof(unsigned char)*img_height*img_width);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }		
	
#if timing
	double accumuTime = 0.0;
	for(int cnt = 0; cnt < 100; cnt++)
	{
#endif	
	std::chrono::high_resolution_clock c;
	std::chrono::high_resolution_clock::time_point start_time = c.now();
	
    // copy host memory to device
	std::cout<<"\n Copy data from the host memory to the CUDA device"<<std::endl;
    err = cudaMemcpy(d_inputData, h_inputData, sizeof(unsigned char)*img_height*img_width, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy (d_inputData, h_inputData) returned error code %d, line(%d)\n", err, __LINE__);
        exit(EXIT_FAILURE);
    }	
	
    // Launch the median filter CUDA Kernel
	const dim3 blockSize(threadsPerBlock, threadsPerBlock);
    int blocksPerGrid = (IMG_SIZE + threadsPerBlock - 1) / threadsPerBlock;
	const dim3 gridSize(blocksPerGrid, blocksPerGrid);	
    printf(" CUDA kernel launch with a grid of (%d * %d) blocks of (%d * %d) threads\n", blocksPerGrid, blocksPerGrid, threadsPerBlock, threadsPerBlock);
    medianFilterGPU<<<gridSize, blockSize>>>(d_inputData, d_outputData, filterSize, img_height, img_width);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch median filter kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf(" Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(outputData_GPU, d_outputData, sizeof(unsigned char)*img_height*img_width, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector outputData_GPU from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	

	std::chrono::high_resolution_clock::time_point end_time = c.now();
	double elapsed_time = (double) std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
	std::cout<<" Time cost in GPU processing: "<<elapsed_time<<"s"<<std::endl;	
	
#if timing
	accumuTime += elapsed_time;
	}
	accumuTime /= 100;
	std::cout<<"\n Averaged copy-compute-copy time is: "<<accumuTime<<"s"<<std::endl;
#endif	
	
    // Verify that the result vector is correct
	std::cout<<" Verification of the results from GPU"<<std::endl;
	float accumuError = 0;
    for (unsigned i = 0; i < img_height; ++i)
	{
		for (unsigned j = 0; j < img_width; ++j)
		{
			double currentError = fabs(outputData_CPU[i * img_width + j] - outputData_GPU[i * img_width + j]);
			if(currentError != 0)
				accumuError += currentError;
			
			if (currentError > 1e-5)
			{
				fprintf(stderr, "Result verification failed at element %d!\n", i * img_width + j);
				exit(EXIT_FAILURE);
			}
		}		
	}
	std::cout<<" Verification passed! The correct rate is: "<<(1.0 - accumuError/img_height/img_width)*100<<"%"<<std::endl;
	
	std::cout<<" Output the filtered image to "<<outputFileName<<std::endl;
	// output the filtered image
	sdkSavePGM<unsigned char> ( outputFileName, outputData_GPU, IMG_SIZE, IMG_SIZE);

    // Free host memory
	delete[] outputData_GPU;
    delete[] h_inputData;
	delete[] outputData_CPU;
	
    // Free device global memory
    err = cudaFree(d_inputData);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device inputData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
    err = cudaFree(d_outputData);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device outputData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	
	
    // Reset the device and exit
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf(" Done\n");
    return 0;	
}







