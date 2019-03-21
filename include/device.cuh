#ifndef _DEVICE_CUH__
#define _DEVICE_CUH__

// GPU function
__global__ void medianFilterGPU(const unsigned char *d_inputData, unsigned char *d_outputData, const int filterSize, const int img_height, const int img_width);

#endif