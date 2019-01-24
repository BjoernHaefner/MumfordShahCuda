#ifndef MUMFORDSHAH_LIB_IMAGE_H_
#define MUMFORDSHAH_LIB_IMAGE_H_

#ifdef USE_CUDA


#include "cuda/image_gpu.cuh"

template<typename T>
using Image = CUDA::ImageGPU<T>;


#else //no cuda support


#include "image_cpu.h"

template<typename T>
using Image = ImageCPU<T>;


#endif //USE_CUDA

#endif //MUMFORDSHAH_LIB_IMAGE_H_
