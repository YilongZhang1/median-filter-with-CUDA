Developing environment:
  - Architecture: x86_64
  - OS: CentOS 7.0
  - GCC: 4.8.5
  - CUDA: 9.1
  - GPU: Tesla K80
  
Note: if you have a different version of CUDA, you need to change the path of /inc in CMakeLists.txt
  - include_directories(/usr/local/cuda-9.1/samples/common/inc)  
  
To compile and run the executable:
  - cmake .
  - make (there is a ptxas warning in compiling, since I used random quickSort in the code, which contains a recursion. It is fine)
  - ./homework_5 <filter size> <input filename> <output filename>
  - the input image can be in .pgm or .ppm extension with size 512*512
  
  
==============================================================================
Timing statistics:
  filterSize                                  3*3      7*7      11*11    15*15
-----------------------------------------------------------------------------------
  - (16 * 16) blocks of (32 * 32) threads     0.29     0.32     0.38     error
  - (32 * 32) blocks of (16 * 16) threads     0.24     0.28     0.34     error
  - (64 * 64) blocks of (8 * 8) threads       0.15     0.17     0.24     0.44