#ifndef _MEDIANFILTER_H__
#define _MEDIANFILTER_H__

#define IMG_SIZE 512	// used to pre-allocate memory for h_inputData
#define threadsPerBlock 8
// CPU function
unsigned char* medianFilterCPU(const unsigned char* h_inputData, const int img_height, const int img_width, const int filterSize);


#endif