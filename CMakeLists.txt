cmake_minimum_required(VERSION 3.5)
project(median_filter_CUDA)

set(CMAKE_C_FLAGS "-std=c11 -Wall -Wextra -Wshadow -Werror")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wshadow -Werror") 
                               
find_package(CUDA QUIET REQUIRED)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)

# include CUDA. Note: you might have to change this path on a different machine
include_directories(/usr/local/cuda-9.1/samples/common/inc)

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(${PROJECT_NAME} src/image_proc.cu src/CPU_code.cpp src/GPU_code.cu)