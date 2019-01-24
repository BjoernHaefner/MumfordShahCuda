/*
 * CUDAKernels.cu
 *
 *  Created on: Jul 12, 2017
 *      Author: haefner
 */

//stl
#include <iostream>

//local cuda code
#include "cuda_common.cuh"
#include "cuda_kernels.cuh"

namespace CUDA {

//############################################################################
template<typename T>
__global__ void abs1DKernel(  T    *arr,
                              int  length )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  T val = arr[x];
  arr[x] = val < (T) 0 ? -val : val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void abs1DKernel(  float   *arr,
int  length );


//############################################################################
template<typename T>
void  abs1D( T   *arr,
             int length )
{
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  abs1DKernel<<< dimGrid, dimBlock >>>( arr, length );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  abs1D( float   *arr,
int length );



//############################################################################
template<typename T>
__global__ void abs2DKernel(  T    *arr,
                              int  width,
                              int  height )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int pos = y*width + x;

  T val = arr[pos];
  arr[pos] = (val < ((T) 0)) ? -val : val;

}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void abs2DKernel(  float   *arr,
int  width,
int  height );


//############################################################################
template<typename T>
void  abs2D( T   *arr,
             int width,
             int  height )
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  abs2DKernel<<< dimGrid, dimBlock >>>( arr, width, height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  abs2D( float   *arr,
int width,
int  height );



//############################################################################
template<typename T>
__global__ void abs3DKernel(  T    *arr,
                              int  width,
                              int  height,
                              int  depth )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int pos = x + y*width + z*width*height;

  T val = arr[pos];
  arr[pos] = (val < ((T) 0)) ? -val : val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void abs3DKernel(  float   *arr,
int  width,
int  height,
int  depth );


//############################################################################
template<typename T>
void  abs3D( T   *arr,
             int  width,
             int  height,
             int  depth )
{
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  abs3DKernel<<< dimGrid, dimBlock >>>( arr, width, height, depth );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  abs3D( float   *arr,
int width,
int  height,
int  depth );



//############################################################################
template<typename T>
__global__ void addToArray1DKernel(  T    *arr,
                                     int  length,
                                     T    val )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  arr[x] += val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray1DKernel(  float   *arr,
int  length,
float    val );


//############################################################################
template<typename T>
void  addToArray1D( T   *arr,
                    int length,
                    T   val )
{
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  addToArray1DKernel<<< dimGrid, dimBlock >>>( arr, length, val );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray1D( float   *arr,
int length,
float   val );



//############################################################################
template<typename T>
__global__ void addToArray2DKernel(  T    *arr,
                                     int  width,
                                     int  height,
                                     T    val )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int pos = y*width + x;

  arr[pos] += val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray2DKernel(  float   *arr,
int  width,
int  height,
float    val );


//############################################################################
template<typename T>
void  addToArray2D( T   *arr,
                    int width,
                    int  height,
                    T   val )
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  addToArray2DKernel<<< dimGrid, dimBlock >>>( arr, width, height, val );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray2D( float   *arr,
int width,
int  height,
float   val );



//############################################################################
template<typename T>
__global__ void addToArray3DKernel(  T    *arr,
                                     int  width,
                                     int  height,
                                     int  depth,
                                     T    val )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int pos = z*width*height + y*width + x;

  arr[pos] += val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray3DKernel(  float   *arr,
int  width,
int  height,
int  depth,
float    val );


//############################################################################
template<typename T>
void  addToArray3D( T   *arr,
                    int width,
                    int  height,
                    int  depth,
                    T   val )
{
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  addToArray3DKernel<<< dimGrid, dimBlock >>>( arr, width, height, depth, val );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray3D( float   *arr,
int width,
int  height,
int  depth,
float   val );



//############################################################################
template<typename T>
__global__ void addToArray1DKernel(  T    *arr1,
                                     int  length,
                                     const T    *arr2 )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  arr1[x] += arr2[x];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray1DKernel(  float   *arr1,
int  length,
const float    *arr2 );


//############################################################################
template<typename T>
void  addToArray1D( T   *arr1,
                    int length,
                    const T   *arr2 )
{
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  addToArray1DKernel<<< dimGrid, dimBlock >>>( arr1, length, arr2 );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray1D( float   *arr1,
int length,
const float   *arr2 );



//############################################################################
template<typename T>
__global__ void addToArray2DKernel(  T    *arr1_2d,
                                     int  width,
                                     int  height,
                                     const T    *arr2_2d )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int pos = y*width + x;

  arr1_2d[pos] += arr2_2d[pos];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray2DKernel(  float   *arr1_2d,
int  width,
int  height,
const float    *arr2_2d );


//############################################################################
template<typename T>
void  addToArray2D( T   *arr1_2d,
                    int width,
                    int  height,
                    const T   *arr2_2d )
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  addToArray2DKernel<<< dimGrid, dimBlock >>>( arr1_2d, width, height, arr2_2d );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray2D( float   *arr1_2d,
int width,
int  height,
const float   *arr2_2d );



//############################################################################
template<typename T>
__global__ void addToArray3DKernel(  T    *arr1_3d,
                                     int  width,
                                     int  height,
                                     int  depth,
                                     const T    *arr2_3d )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int pos = z*width*height + y*width + x;

  arr1_3d[pos] += arr2_3d[pos];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void addToArray3DKernel(  float   *arr1_3d,
int  width,
int  height,
int  depth,
const float    *arr2_3d );


//############################################################################
template<typename T>
void  addToArray3D( T   *arr1_3d,
                    int width,
                    int  height,
                    int  depth,
                    const T   *arr2_3d )
{
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  addToArray3DKernel<<< dimGrid, dimBlock >>>( arr1_3d, width, height, depth, arr2_3d );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  addToArray3D( float   *arr1_3d,
int width,
int  height,
int  depth,
const float   *arr2_3d );



//############################################################################
template<typename T>
__global__ void subFromArray1DKernel(  T    *arr1,
                                       int  length,
                                       const T    *arr2 )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  arr1[x] -= arr2[x];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void subFromArray1DKernel(  float   *arr1,
int  length,
const float    *arr2 );


//############################################################################
template<typename T>
void  subFromArray1D( T   *arr1,
                      int length,
                      const T   *arr2 )
{
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  subFromArray1DKernel<<< dimGrid, dimBlock >>>( arr1, length, arr2 );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  subFromArray1D( float   *arr1,
int length,
const float   *arr2 );



//############################################################################
template<typename T>
__global__ void subFromArray2DKernel(  T    *arr1_2d,
                                       int  width,
                                       int  height,
                                       const T    *arr2_2d )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int pos = y*width + x;

  arr1_2d[pos] -= arr2_2d[pos];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void subFromArray2DKernel(  float   *arr1_2d,
int  width,
int  height,
const float    *arr2_2d );


//############################################################################
template<typename T>
void  subFromArray2D( T   *arr1_2d,
                      int width,
                      int  height,
                      const T   *arr2_2d )
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  subFromArray2DKernel<<< dimGrid, dimBlock >>>( arr1_2d, width, height, arr2_2d );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  subFromArray2D( float   *arr1_2d,
int width,
int  height,
const float   *arr2_2d );



//############################################################################
template<typename T>
__global__ void subFromArray3DKernel(  T    *arr1_3d,
                                       int  width,
                                       int  height,
                                       int  depth,
                                       const T    *arr2_3d )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int pos = z*width*height + y*width + x;

  arr1_3d[pos] -= arr2_3d[pos];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void subFromArray3DKernel(  float   *arr1_3d,
int  width,
int  height,
int  depth,
const float    *arr2_3d );


//############################################################################
template<typename T>
void  subFromArray3D( T   *arr1_3d,
                      int width,
                      int  height,
                      int  depth,
                      const T   *arr2_3d )
{
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  subFromArray3DKernel<<< dimGrid, dimBlock >>>( arr1_3d, width, height, depth, arr2_3d );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template    void  subFromArray3D( float   *arr1_3d,
int width,
int  height,
int  depth,
const float   *arr2_3d );



//############################################################################
template< typename T >
__global__ void alignDataCUDA2MatlabKernel( const T *     arr_cuda,
                                            size_t  width,
                                            size_t  height,
                                            size_t  channels,
                                            T *     arr_matlab )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  for (int c = 0; c<channels; c++)
  {
    int lin_matlab  = y + x * height  + c * width * height;
    int lin_cuda    = x + y * width   + c * width * height;

    arr_matlab[lin_matlab] = arr_cuda[lin_cuda];
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void alignDataCUDA2MatlabKernel(  const float * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * arr_matlab );
template __global__ void alignDataCUDA2MatlabKernel(  const bool * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
bool * arr_matlab );



//############################################################################
template< typename T >
void alignDataCUDA2Matlab( const T * arr_cuda,
                           size_t width,
                           size_t height,
                           size_t channels,
                           T * arr_matlab)
{
  dim3 dimGrid, dimBlock;
  CUDA::get2DGridBlock(width, height, dimGrid, dimBlock);
  CUDA::alignDataCUDA2MatlabKernel<<<dimGrid, dimBlock>>>(arr_cuda, width, height, channels, arr_matlab);

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

}//############################################################################
// Explicit instantiations:
template void alignDataCUDA2Matlab(  const float * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * arr_matlab );
template void alignDataCUDA2Matlab(  const bool * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
bool * arr_matlab );



//############################################################################
template< typename T1, typename T2 >
__global__ void alignDataCUDA2OpCVKernel( const T1 *     arr_cuda,
                                          size_t  width,
                                          size_t  height,
                                          size_t  channels,
                                          T2 *     arr_opcv,
                                          size_t  step_y_opcv,
                                          size_t  step_x_opcv )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  for (int c = 0; c<channels; c++)
  {
    int lin_opcv = y * step_y_opcv + x * step_x_opcv + c;
    int lin_cuda = c * width * height + y * width + x;

    arr_opcv[lin_opcv]= (T1) arr_cuda[lin_cuda];
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void alignDataCUDA2OpCVKernel(  const float * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template __global__ void alignDataCUDA2OpCVKernel(  const bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
bool * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template __global__ void alignDataCUDA2OpCVKernel(  const bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );



//############################################################################
template< typename T1, typename T2 >
void alignDataCUDA2OpCV( const T1 *     dvc_cuda,
                         size_t  width,
                         size_t  height,
                         size_t  channels,
                         T2 * dvc_opcv,
                         size_t  step_y,
                         size_t  step_x )
{
  dim3 dimGrid, dimBlock;
  CUDA::get2DGridBlock(width, height, dimGrid, dimBlock);
  CUDA::alignDataCUDA2OpCVKernel<<<dimGrid, dimBlock>>>(dvc_cuda, width, height, channels, dvc_opcv, step_y, step_x);

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

}//############################################################################
// Explicit instantiations:
template void alignDataCUDA2OpCV(  const float * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template void alignDataCUDA2OpCV(  const bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
bool * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template void alignDataCUDA2OpCV(  const bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );



//############################################################################
template< typename T >
__global__ void alignDataMatlab2CUDAKernel( T *     arr_cuda,
                                            size_t  width,
                                            size_t  height,
                                            size_t  channels,
                                            const T *     arr_matlab )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  for (int c = 0; c < channels; c++)
  {
    size_t lin_matlab  = y + x * height + c * width * height;
    size_t lin_cuda    = x + y * width  + c * width * height;

    //      if (x == 239 && y == 210) printf("%dx%d: %f", x, y, arr_matlab[lin_matlab]);

    arr_cuda[lin_cuda] = arr_matlab[lin_matlab];
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void alignDataMatlab2CUDAKernel(  float * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * arr_matlab );
template __global__ void alignDataMatlab2CUDAKernel(  bool * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
const bool * arr_matlab );



//############################################################################
template< typename T >
void alignDataMatlab2CUDA( T * arr_cuda,
                           size_t width,
                           size_t height,
                           size_t channels,
                           const T * arr_matlab)
{
  dim3 dimGrid, dimBlock;
  CUDA::get2DGridBlock(width, height, dimGrid, dimBlock);
  //    std::cout << "jump in alignDataMatlab2CUDAKernel with wxhxc: " << width << "x" << height <<"x"<<channels << std::endl;
  //    std::cout << "dimGrid: " << dimGrid.x << "x" << dimGrid.y <<"x"<<dimGrid.z << std::endl;
  //    std::cout << "dimBlock: " << dimBlock.x << "x" << dimBlock.y <<"x"<<dimBlock.z << std::endl;
  CUDA::alignDataMatlab2CUDAKernel<<<dimGrid, dimBlock>>>(arr_cuda, width, height, channels, arr_matlab);

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

}//############################################################################
// Explicit instantiations:
template void alignDataMatlab2CUDA(  float * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * arr_matlab );
template void alignDataMatlab2CUDA(  bool * arr_cuda,
size_t  width,
size_t  height,
size_t  channels,
const bool * arr_matlab );



//############################################################################
template< typename T1, typename T2 >
__global__ void alignDataOpCV2CUDAKernel( T1 *     arr_cuda,
                                          size_t  width,
                                          size_t  height,
                                          size_t  channels,
                                          const T2 *     arr_opcv,
                                          size_t  step_y_opcv,
                                          size_t  step_x_opcv )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  for (int c = 0; c<channels; c++)
  {
    int lin_opcv = y * step_y_opcv + x * step_x_opcv + c;
    int lin_cuda = c * width * height + y * width + x;

    arr_cuda[lin_cuda] = (T1) arr_opcv[lin_opcv];
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void alignDataOpCV2CUDAKernel(  float * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template __global__ void alignDataOpCV2CUDAKernel(  bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const bool * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template __global__ void alignDataOpCV2CUDAKernel(  bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );



//############################################################################
template< typename T1, typename T2 >
void alignDataOpCV2CUDA( T1 *     dvc_cuda,
                         size_t  width,
                         size_t  height,
                         size_t  channels,
                         const T2 * dvc_opcv,
                         size_t  step_y,
                         size_t  step_x )
{
  dim3 dimGrid, dimBlock;
  CUDA::get2DGridBlock(width, height, dimGrid, dimBlock);
  CUDA::alignDataOpCV2CUDAKernel<<<dimGrid, dimBlock>>>(dvc_cuda, width, height, channels, dvc_opcv, step_y, step_x);

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

}//############################################################################
// Explicit instantiations:
template void alignDataOpCV2CUDA(  float * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template void alignDataOpCV2CUDA(  bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const bool * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );
template void alignDataOpCV2CUDA(  bool * dvc_cuda,
size_t  width,
size_t  height,
size_t  channels,
const float * dvc_opcv,
size_t  step_y_opcv,
size_t  step_x_opcv );



//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask1ChannelKernel( T1 *         ptr_img,
                                           int            width,
                                           int            height,
                                           const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  int     pos     = y * width + x;

  if ( ptr_mask[pos] == (T2)0 )
    ptr_img[pos] = (T1)0;
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask1ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask1ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask3ChannelKernel( T1 *         ptr_img,
                                           int            width,
                                           int            height,
                                           const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  int     pos0     = y * width + x;
  int     pos1     = pos0 + width*height;
  int     pos2     = pos1 + width*height;

  if ( ptr_mask[pos0] == (T2) 0 )
  {
    ptr_img[pos0] = (T1) 0;
    ptr_img[pos1] = (T1) 0;
    ptr_img[pos2] = (T1) 0;
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask3ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask3ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask3ChannelsKernel( T1 *         ptr_img,
                                            int            width,
                                            int            height,
                                            int            channels,
                                            const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  size_t     img_size = width*height;

  size_t     pos0     = y * width + x;
  size_t     pos1     = pos0 + img_size;
  size_t     pos2     = pos1 + img_size;

  size_t step3imgs = 3*img_size;

  if ( ptr_mask[pos0] == (T2) 0 )
  {
    for (size_t c = 0; c < channels/3; c++)
    {
      ptr_img[pos0 + c*step3imgs] = (T1) 0;
      ptr_img[pos1 + c*step3imgs] = (T1) 0;
      ptr_img[pos2 + c*step3imgs] = (T1) 0;
    }
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask3ChannelsKernel( float*         ptr_img,
int            width,
int            height,
int            channels,
bool  const*   ptr_mask );
template __global__ void  apply2DMask3ChannelsKernel( float*         ptr_img,
int            width,
int            height,
int            channels,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask4ChannelKernel( T1 *         ptr_img,
                                           int            width,
                                           int            height,
                                           const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  size_t     img_size = width*height;

  size_t     pos0     = y * width + x;

  if ( ptr_mask[pos0] == (T2) 0 )
  {
    ptr_img[pos0             ] = (T1) 0;
    ptr_img[pos0 + 1*img_size] = (T1) 0;
    ptr_img[pos0 + 2*img_size] = (T1) 0;
    ptr_img[pos0 + 3*img_size] = (T1) 0;
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask4ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask4ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask16ChannelKernel( T1 *         ptr_img,
                                            int            width,
                                            int            height,
                                            const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  size_t     img_size = width*height;

  size_t     pos0     = y * width + x;

  if ( ptr_mask[pos0] == (T2) 0 )
  {
    ptr_img[pos0             ] = (T1) 0;
    ptr_img[pos0 + 1*img_size] = (T1) 0;
    ptr_img[pos0 + 2*img_size] = (T1) 0;
    ptr_img[pos0 + 3*img_size] = (T1) 0;
    ptr_img[pos0 + 4*img_size] = (T1) 0;
    ptr_img[pos0 + 5*img_size] = (T1) 0;
    ptr_img[pos0 + 6*img_size] = (T1) 0;
    ptr_img[pos0 + 7*img_size] = (T1) 0;
    ptr_img[pos0 + 8*img_size] = (T1) 0;
    ptr_img[pos0 + 9*img_size] = (T1) 0;
    ptr_img[pos0 + 10*img_size] = (T1) 0;
    ptr_img[pos0 + 11*img_size] = (T1) 0;
    ptr_img[pos0 + 12*img_size] = (T1) 0;
    ptr_img[pos0 + 13*img_size] = (T1) 0;
    ptr_img[pos0 + 14*img_size] = (T1) 0;
    ptr_img[pos0 + 15*img_size] = (T1) 0;

  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask16ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask16ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask64ChannelKernel( T1 *         ptr_img,
                                            int            width,
                                            int            height,
                                            const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  size_t     img_size = width*height;

  size_t pos0 = y * width + x;

  if ( ptr_mask[pos0] == (T2) 0 )
  {
    ptr_img[pos0             ] = (T1) 0;
    ptr_img[pos0 + 1*img_size] = (T1) 0;
    ptr_img[pos0 + 2*img_size] = (T1) 0;
    ptr_img[pos0 + 3*img_size] = (T1) 0;
    ptr_img[pos0 + 4*img_size] = (T1) 0;
    ptr_img[pos0 + 5*img_size] = (T1) 0;
    ptr_img[pos0 + 6*img_size] = (T1) 0;
    ptr_img[pos0 + 7*img_size] = (T1) 0;
    ptr_img[pos0 + 8*img_size] = (T1) 0;
    ptr_img[pos0 + 9*img_size] = (T1) 0;
    ptr_img[pos0 + 10*img_size] = (T1) 0;
    ptr_img[pos0 + 11*img_size] = (T1) 0;
    ptr_img[pos0 + 12*img_size] = (T1) 0;
    ptr_img[pos0 + 13*img_size] = (T1) 0;
    ptr_img[pos0 + 14*img_size] = (T1) 0;
    ptr_img[pos0 + 15*img_size] = (T1) 0;
    ptr_img[pos0 + 16*img_size] = (T1) 0;
    ptr_img[pos0 + 17*img_size] = (T1) 0;
    ptr_img[pos0 + 18*img_size] = (T1) 0;
    ptr_img[pos0 + 19*img_size] = (T1) 0;
    ptr_img[pos0 + 20*img_size] = (T1) 0;
    ptr_img[pos0 + 21*img_size] = (T1) 0;
    ptr_img[pos0 + 22*img_size] = (T1) 0;
    ptr_img[pos0 + 23*img_size] = (T1) 0;
    ptr_img[pos0 + 24*img_size] = (T1) 0;
    ptr_img[pos0 + 25*img_size] = (T1) 0;
    ptr_img[pos0 + 26*img_size] = (T1) 0;
    ptr_img[pos0 + 27*img_size] = (T1) 0;
    ptr_img[pos0 + 28*img_size] = (T1) 0;
    ptr_img[pos0 + 29*img_size] = (T1) 0;
    ptr_img[pos0 + 30*img_size] = (T1) 0;
    ptr_img[pos0 + 31*img_size] = (T1) 0;
    ptr_img[pos0 + 32*img_size] = (T1) 0;
    ptr_img[pos0 + 33*img_size] = (T1) 0;
    ptr_img[pos0 + 34*img_size] = (T1) 0;
    ptr_img[pos0 + 35*img_size] = (T1) 0;
    ptr_img[pos0 + 36*img_size] = (T1) 0;
    ptr_img[pos0 + 37*img_size] = (T1) 0;
    ptr_img[pos0 + 38*img_size] = (T1) 0;
    ptr_img[pos0 + 39*img_size] = (T1) 0;
    ptr_img[pos0 + 40*img_size] = (T1) 0;
    ptr_img[pos0 + 41*img_size] = (T1) 0;
    ptr_img[pos0 + 42*img_size] = (T1) 0;
    ptr_img[pos0 + 43*img_size] = (T1) 0;
    ptr_img[pos0 + 44*img_size] = (T1) 0;
    ptr_img[pos0 + 45*img_size] = (T1) 0;
    ptr_img[pos0 + 46*img_size] = (T1) 0;
    ptr_img[pos0 + 47*img_size] = (T1) 0;
    ptr_img[pos0 + 48*img_size] = (T1) 0;
    ptr_img[pos0 + 49*img_size] = (T1) 0;
    ptr_img[pos0 + 50*img_size] = (T1) 0;
    ptr_img[pos0 + 51*img_size] = (T1) 0;
    ptr_img[pos0 + 52*img_size] = (T1) 0;
    ptr_img[pos0 + 53*img_size] = (T1) 0;
    ptr_img[pos0 + 54*img_size] = (T1) 0;
    ptr_img[pos0 + 55*img_size] = (T1) 0;
    ptr_img[pos0 + 56*img_size] = (T1) 0;
    ptr_img[pos0 + 57*img_size] = (T1) 0;
    ptr_img[pos0 + 58*img_size] = (T1) 0;
    ptr_img[pos0 + 59*img_size] = (T1) 0;
    ptr_img[pos0 + 60*img_size] = (T1) 0;
    ptr_img[pos0 + 61*img_size] = (T1) 0;
    ptr_img[pos0 + 62*img_size] = (T1) 0;
    ptr_img[pos0 + 63*img_size] = (T1) 0;
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask64ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask64ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );





//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMask256ChannelKernel( T1 *         ptr_img,
                                             int            width,
                                             int            height,
                                             const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  size_t  img_size = width*height;

  size_t  pos0 = y * width + x;


  if ( ptr_mask[pos0] == (T2) 0 )
  {
    ptr_img[pos0             ] = (T1) 0;
    ptr_img[pos0 + 1*img_size] = (T1) 0;
    ptr_img[pos0 + 2*img_size] = (T1) 0;
    ptr_img[pos0 + 3*img_size] = (T1) 0;
    ptr_img[pos0 + 4*img_size] = (T1) 0;
    ptr_img[pos0 + 5*img_size] = (T1) 0;
    ptr_img[pos0 + 6*img_size] = (T1) 0;
    ptr_img[pos0 + 7*img_size] = (T1) 0;
    ptr_img[pos0 + 8*img_size] = (T1) 0;
    ptr_img[pos0 + 9*img_size] = (T1) 0;
    ptr_img[pos0 + 10*img_size] = (T1) 0;
    ptr_img[pos0 + 11*img_size] = (T1) 0;
    ptr_img[pos0 + 12*img_size] = (T1) 0;
    ptr_img[pos0 + 13*img_size] = (T1) 0;
    ptr_img[pos0 + 14*img_size] = (T1) 0;
    ptr_img[pos0 + 15*img_size] = (T1) 0;
    ptr_img[pos0 + 16*img_size] = (T1) 0;
    ptr_img[pos0 + 17*img_size] = (T1) 0;
    ptr_img[pos0 + 18*img_size] = (T1) 0;
    ptr_img[pos0 + 19*img_size] = (T1) 0;
    ptr_img[pos0 + 20*img_size] = (T1) 0;
    ptr_img[pos0 + 21*img_size] = (T1) 0;
    ptr_img[pos0 + 22*img_size] = (T1) 0;
    ptr_img[pos0 + 23*img_size] = (T1) 0;
    ptr_img[pos0 + 24*img_size] = (T1) 0;
    ptr_img[pos0 + 25*img_size] = (T1) 0;
    ptr_img[pos0 + 26*img_size] = (T1) 0;
    ptr_img[pos0 + 27*img_size] = (T1) 0;
    ptr_img[pos0 + 28*img_size] = (T1) 0;
    ptr_img[pos0 + 29*img_size] = (T1) 0;
    ptr_img[pos0 + 30*img_size] = (T1) 0;
    ptr_img[pos0 + 31*img_size] = (T1) 0;
    ptr_img[pos0 + 32*img_size] = (T1) 0;
    ptr_img[pos0 + 33*img_size] = (T1) 0;
    ptr_img[pos0 + 34*img_size] = (T1) 0;
    ptr_img[pos0 + 35*img_size] = (T1) 0;
    ptr_img[pos0 + 36*img_size] = (T1) 0;
    ptr_img[pos0 + 37*img_size] = (T1) 0;
    ptr_img[pos0 + 38*img_size] = (T1) 0;
    ptr_img[pos0 + 39*img_size] = (T1) 0;
    ptr_img[pos0 + 40*img_size] = (T1) 0;
    ptr_img[pos0 + 41*img_size] = (T1) 0;
    ptr_img[pos0 + 42*img_size] = (T1) 0;
    ptr_img[pos0 + 43*img_size] = (T1) 0;
    ptr_img[pos0 + 44*img_size] = (T1) 0;
    ptr_img[pos0 + 45*img_size] = (T1) 0;
    ptr_img[pos0 + 46*img_size] = (T1) 0;
    ptr_img[pos0 + 47*img_size] = (T1) 0;
    ptr_img[pos0 + 48*img_size] = (T1) 0;
    ptr_img[pos0 + 49*img_size] = (T1) 0;
    ptr_img[pos0 + 50*img_size] = (T1) 0;
    ptr_img[pos0 + 51*img_size] = (T1) 0;
    ptr_img[pos0 + 52*img_size] = (T1) 0;
    ptr_img[pos0 + 53*img_size] = (T1) 0;
    ptr_img[pos0 + 54*img_size] = (T1) 0;
    ptr_img[pos0 + 55*img_size] = (T1) 0;
    ptr_img[pos0 + 56*img_size] = (T1) 0;
    ptr_img[pos0 + 57*img_size] = (T1) 0;
    ptr_img[pos0 + 58*img_size] = (T1) 0;
    ptr_img[pos0 + 59*img_size] = (T1) 0;
    ptr_img[pos0 + 60*img_size] = (T1) 0;
    ptr_img[pos0 + 61*img_size] = (T1) 0;
    ptr_img[pos0 + 62*img_size] = (T1) 0;
    ptr_img[pos0 + 63*img_size] = (T1) 0;
    ptr_img[pos0 + 64*img_size] = (T1) 0;
    ptr_img[pos0 + 65*img_size] = (T1) 0;
    ptr_img[pos0 + 66*img_size] = (T1) 0;
    ptr_img[pos0 + 67*img_size] = (T1) 0;
    ptr_img[pos0 + 68*img_size] = (T1) 0;
    ptr_img[pos0 + 69*img_size] = (T1) 0;
    ptr_img[pos0 + 70*img_size] = (T1) 0;
    ptr_img[pos0 + 71*img_size] = (T1) 0;
    ptr_img[pos0 + 72*img_size] = (T1) 0;
    ptr_img[pos0 + 73*img_size] = (T1) 0;
    ptr_img[pos0 + 74*img_size] = (T1) 0;
    ptr_img[pos0 + 75*img_size] = (T1) 0;
    ptr_img[pos0 + 76*img_size] = (T1) 0;
    ptr_img[pos0 + 77*img_size] = (T1) 0;
    ptr_img[pos0 + 78*img_size] = (T1) 0;
    ptr_img[pos0 + 79*img_size] = (T1) 0;
    ptr_img[pos0 + 80*img_size] = (T1) 0;
    ptr_img[pos0 + 81*img_size] = (T1) 0;
    ptr_img[pos0 + 82*img_size] = (T1) 0;
    ptr_img[pos0 + 83*img_size] = (T1) 0;
    ptr_img[pos0 + 84*img_size] = (T1) 0;
    ptr_img[pos0 + 85*img_size] = (T1) 0;
    ptr_img[pos0 + 86*img_size] = (T1) 0;
    ptr_img[pos0 + 87*img_size] = (T1) 0;
    ptr_img[pos0 + 88*img_size] = (T1) 0;
    ptr_img[pos0 + 89*img_size] = (T1) 0;
    ptr_img[pos0 + 90*img_size] = (T1) 0;
    ptr_img[pos0 + 91*img_size] = (T1) 0;
    ptr_img[pos0 + 92*img_size] = (T1) 0;
    ptr_img[pos0 + 93*img_size] = (T1) 0;
    ptr_img[pos0 + 94*img_size] = (T1) 0;
    ptr_img[pos0 + 95*img_size] = (T1) 0;
    ptr_img[pos0 + 96*img_size] = (T1) 0;
    ptr_img[pos0 + 97*img_size] = (T1) 0;
    ptr_img[pos0 + 98*img_size] = (T1) 0;
    ptr_img[pos0 + 99*img_size] = (T1) 0;
    ptr_img[pos0 + 100*img_size] = (T1) 0;
    ptr_img[pos0 + 101*img_size] = (T1) 0;
    ptr_img[pos0 + 102*img_size] = (T1) 0;
    ptr_img[pos0 + 103*img_size] = (T1) 0;
    ptr_img[pos0 + 104*img_size] = (T1) 0;
    ptr_img[pos0 + 105*img_size] = (T1) 0;
    ptr_img[pos0 + 106*img_size] = (T1) 0;
    ptr_img[pos0 + 107*img_size] = (T1) 0;
    ptr_img[pos0 + 108*img_size] = (T1) 0;
    ptr_img[pos0 + 109*img_size] = (T1) 0;
    ptr_img[pos0 + 110*img_size] = (T1) 0;
    ptr_img[pos0 + 111*img_size] = (T1) 0;
    ptr_img[pos0 + 112*img_size] = (T1) 0;
    ptr_img[pos0 + 113*img_size] = (T1) 0;
    ptr_img[pos0 + 114*img_size] = (T1) 0;
    ptr_img[pos0 + 115*img_size] = (T1) 0;
    ptr_img[pos0 + 116*img_size] = (T1) 0;
    ptr_img[pos0 + 117*img_size] = (T1) 0;
    ptr_img[pos0 + 118*img_size] = (T1) 0;
    ptr_img[pos0 + 119*img_size] = (T1) 0;
    ptr_img[pos0 + 120*img_size] = (T1) 0;
    ptr_img[pos0 + 121*img_size] = (T1) 0;
    ptr_img[pos0 + 122*img_size] = (T1) 0;
    ptr_img[pos0 + 123*img_size] = (T1) 0;
    ptr_img[pos0 + 124*img_size] = (T1) 0;
    ptr_img[pos0 + 125*img_size] = (T1) 0;
    ptr_img[pos0 + 126*img_size] = (T1) 0;
    ptr_img[pos0 + 127*img_size] = (T1) 0;
    ptr_img[pos0 + 128*img_size] = (T1) 0;
    ptr_img[pos0 + 129*img_size] = (T1) 0;
    ptr_img[pos0 + 130*img_size] = (T1) 0;
    ptr_img[pos0 + 131*img_size] = (T1) 0;
    ptr_img[pos0 + 132*img_size] = (T1) 0;
    ptr_img[pos0 + 133*img_size] = (T1) 0;
    ptr_img[pos0 + 134*img_size] = (T1) 0;
    ptr_img[pos0 + 135*img_size] = (T1) 0;
    ptr_img[pos0 + 136*img_size] = (T1) 0;
    ptr_img[pos0 + 137*img_size] = (T1) 0;
    ptr_img[pos0 + 138*img_size] = (T1) 0;
    ptr_img[pos0 + 139*img_size] = (T1) 0;
    ptr_img[pos0 + 140*img_size] = (T1) 0;
    ptr_img[pos0 + 141*img_size] = (T1) 0;
    ptr_img[pos0 + 142*img_size] = (T1) 0;
    ptr_img[pos0 + 143*img_size] = (T1) 0;
    ptr_img[pos0 + 144*img_size] = (T1) 0;
    ptr_img[pos0 + 145*img_size] = (T1) 0;
    ptr_img[pos0 + 146*img_size] = (T1) 0;
    ptr_img[pos0 + 147*img_size] = (T1) 0;
    ptr_img[pos0 + 148*img_size] = (T1) 0;
    ptr_img[pos0 + 149*img_size] = (T1) 0;
    ptr_img[pos0 + 150*img_size] = (T1) 0;
    ptr_img[pos0 + 151*img_size] = (T1) 0;
    ptr_img[pos0 + 152*img_size] = (T1) 0;
    ptr_img[pos0 + 153*img_size] = (T1) 0;
    ptr_img[pos0 + 154*img_size] = (T1) 0;
    ptr_img[pos0 + 155*img_size] = (T1) 0;
    ptr_img[pos0 + 156*img_size] = (T1) 0;
    ptr_img[pos0 + 157*img_size] = (T1) 0;
    ptr_img[pos0 + 158*img_size] = (T1) 0;
    ptr_img[pos0 + 159*img_size] = (T1) 0;
    ptr_img[pos0 + 160*img_size] = (T1) 0;
    ptr_img[pos0 + 161*img_size] = (T1) 0;
    ptr_img[pos0 + 162*img_size] = (T1) 0;
    ptr_img[pos0 + 163*img_size] = (T1) 0;
    ptr_img[pos0 + 164*img_size] = (T1) 0;
    ptr_img[pos0 + 165*img_size] = (T1) 0;
    ptr_img[pos0 + 166*img_size] = (T1) 0;
    ptr_img[pos0 + 167*img_size] = (T1) 0;
    ptr_img[pos0 + 168*img_size] = (T1) 0;
    ptr_img[pos0 + 169*img_size] = (T1) 0;
    ptr_img[pos0 + 170*img_size] = (T1) 0;
    ptr_img[pos0 + 171*img_size] = (T1) 0;
    ptr_img[pos0 + 172*img_size] = (T1) 0;
    ptr_img[pos0 + 173*img_size] = (T1) 0;
    ptr_img[pos0 + 174*img_size] = (T1) 0;
    ptr_img[pos0 + 175*img_size] = (T1) 0;
    ptr_img[pos0 + 176*img_size] = (T1) 0;
    ptr_img[pos0 + 177*img_size] = (T1) 0;
    ptr_img[pos0 + 178*img_size] = (T1) 0;
    ptr_img[pos0 + 179*img_size] = (T1) 0;
    ptr_img[pos0 + 180*img_size] = (T1) 0;
    ptr_img[pos0 + 181*img_size] = (T1) 0;
    ptr_img[pos0 + 182*img_size] = (T1) 0;
    ptr_img[pos0 + 183*img_size] = (T1) 0;
    ptr_img[pos0 + 184*img_size] = (T1) 0;
    ptr_img[pos0 + 185*img_size] = (T1) 0;
    ptr_img[pos0 + 186*img_size] = (T1) 0;
    ptr_img[pos0 + 187*img_size] = (T1) 0;
    ptr_img[pos0 + 188*img_size] = (T1) 0;
    ptr_img[pos0 + 189*img_size] = (T1) 0;
    ptr_img[pos0 + 190*img_size] = (T1) 0;
    ptr_img[pos0 + 191*img_size] = (T1) 0;
    ptr_img[pos0 + 192*img_size] = (T1) 0;
    ptr_img[pos0 + 193*img_size] = (T1) 0;
    ptr_img[pos0 + 194*img_size] = (T1) 0;
    ptr_img[pos0 + 195*img_size] = (T1) 0;
    ptr_img[pos0 + 196*img_size] = (T1) 0;
    ptr_img[pos0 + 197*img_size] = (T1) 0;
    ptr_img[pos0 + 198*img_size] = (T1) 0;
    ptr_img[pos0 + 199*img_size] = (T1) 0;
    ptr_img[pos0 + 200*img_size] = (T1) 0;
    ptr_img[pos0 + 201*img_size] = (T1) 0;
    ptr_img[pos0 + 202*img_size] = (T1) 0;
    ptr_img[pos0 + 203*img_size] = (T1) 0;
    ptr_img[pos0 + 204*img_size] = (T1) 0;
    ptr_img[pos0 + 205*img_size] = (T1) 0;
    ptr_img[pos0 + 206*img_size] = (T1) 0;
    ptr_img[pos0 + 207*img_size] = (T1) 0;
    ptr_img[pos0 + 208*img_size] = (T1) 0;
    ptr_img[pos0 + 209*img_size] = (T1) 0;
    ptr_img[pos0 + 210*img_size] = (T1) 0;
    ptr_img[pos0 + 211*img_size] = (T1) 0;
    ptr_img[pos0 + 212*img_size] = (T1) 0;
    ptr_img[pos0 + 213*img_size] = (T1) 0;
    ptr_img[pos0 + 214*img_size] = (T1) 0;
    ptr_img[pos0 + 215*img_size] = (T1) 0;
    ptr_img[pos0 + 216*img_size] = (T1) 0;
    ptr_img[pos0 + 217*img_size] = (T1) 0;
    ptr_img[pos0 + 218*img_size] = (T1) 0;
    ptr_img[pos0 + 219*img_size] = (T1) 0;
    ptr_img[pos0 + 220*img_size] = (T1) 0;
    ptr_img[pos0 + 221*img_size] = (T1) 0;
    ptr_img[pos0 + 222*img_size] = (T1) 0;
    ptr_img[pos0 + 223*img_size] = (T1) 0;
    ptr_img[pos0 + 224*img_size] = (T1) 0;
    ptr_img[pos0 + 225*img_size] = (T1) 0;
    ptr_img[pos0 + 226*img_size] = (T1) 0;
    ptr_img[pos0 + 227*img_size] = (T1) 0;
    ptr_img[pos0 + 228*img_size] = (T1) 0;
    ptr_img[pos0 + 229*img_size] = (T1) 0;
    ptr_img[pos0 + 230*img_size] = (T1) 0;
    ptr_img[pos0 + 231*img_size] = (T1) 0;
    ptr_img[pos0 + 232*img_size] = (T1) 0;
    ptr_img[pos0 + 233*img_size] = (T1) 0;
    ptr_img[pos0 + 234*img_size] = (T1) 0;
    ptr_img[pos0 + 235*img_size] = (T1) 0;
    ptr_img[pos0 + 236*img_size] = (T1) 0;
    ptr_img[pos0 + 237*img_size] = (T1) 0;
    ptr_img[pos0 + 238*img_size] = (T1) 0;
    ptr_img[pos0 + 239*img_size] = (T1) 0;
    ptr_img[pos0 + 240*img_size] = (T1) 0;
    ptr_img[pos0 + 241*img_size] = (T1) 0;
    ptr_img[pos0 + 242*img_size] = (T1) 0;
    ptr_img[pos0 + 243*img_size] = (T1) 0;
    ptr_img[pos0 + 244*img_size] = (T1) 0;
    ptr_img[pos0 + 245*img_size] = (T1) 0;
    ptr_img[pos0 + 246*img_size] = (T1) 0;
    ptr_img[pos0 + 247*img_size] = (T1) 0;
    ptr_img[pos0 + 248*img_size] = (T1) 0;
    ptr_img[pos0 + 249*img_size] = (T1) 0;
    ptr_img[pos0 + 250*img_size] = (T1) 0;
    ptr_img[pos0 + 251*img_size] = (T1) 0;
    ptr_img[pos0 + 252*img_size] = (T1) 0;
    ptr_img[pos0 + 253*img_size] = (T1) 0;
    ptr_img[pos0 + 254*img_size] = (T1) 0;
    ptr_img[pos0 + 255*img_size] = (T1) 0;
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMask256ChannelKernel( float*         ptr_img,
int            width,
int            height,
bool  const*   ptr_mask );
template __global__ void  apply2DMask256ChannelKernel( float*         ptr_img,
int            width,
int            height,
float  const*   ptr_mask );




//############################################################################
template< typename T1, typename T2 >
__global__ void apply2DMaskGenericKernel( T1 *         ptr_img,
                                          int            width,
                                          int            height,
                                          int            channels,
                                          const T2  *   ptr_mask )
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;

  int     pos     = y * width + x;

  if ( ptr_mask[pos] == (T2) 0 )
  {
    for ( int c=0; c<channels; ++c )
      ptr_img[ c*width*height + pos ] = 0;
  }
}
//############################################################################
// Explicit instantiations:
template __global__ void  apply2DMaskGenericKernel( float*         ptr_img,
int            width,
int            height,
int            channels,
bool  const*   ptr_mask );
template __global__ void  apply2DMaskGenericKernel( float*         ptr_img,
int            width,
int            height,
int            channels,
float  const*   ptr_mask );



//############################################################################
template< typename T1, typename T2 >
void apply2DMask( T1*         ptr_img,
                  int            width,
                  int            height,
                  int            channels,
                  T2  const*   ptr_mask )
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  if (channels == 1)//one channel
  {
    apply2DMask1ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else if (channels == 3)//rgb image
  {
    apply2DMask3ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else if (channels % 3 == 0)//rgb images
  {
    apply2DMask3ChannelsKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, channels, ptr_mask);
  }
  else if (channels == 4)//four channels (sr with scaling 2)
  {
    apply2DMask4ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else if (channels == 16)//16 channels (sr with scaling 4)
  {
    apply2DMask16ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else if (channels == 64)//64 channels (sr with scaling 8)
  {
    apply2DMask64ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else if (channels == 256)//256 channels (sr with scaling 16)
  {
    apply2DMask256ChannelKernel<<<dimGrid, dimBlock>>>(ptr_img, width, height, ptr_mask);
  }
  else//generic kernel
  {
    apply2DMaskGenericKernel<<< dimGrid, dimBlock >>>( ptr_img, width, height, channels, ptr_mask );
  }

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//############################################################################
// Explicit instantiations:
template void  apply2DMask( float*         ptr_img,
int            width,
int            height,
int            channels,
bool  const*   ptr_mask );
template void  apply2DMask( float*         ptr_img,
int            width,
int            height,
int            channels,
float  const*   ptr_mask );

//############################################################################
template< typename T>
__device__ void  backProjectPixelKernel( const T &depth,
                                         const int &x,
                                         const int &y,
                                         T &X,
                                         T &Y,
                                         T &Z,
                                         const T       *K_gpu )
{
  // Write 3D points as 3 image channels
  X = ( depth * ( (float)x - K_gpu[2] ) ) / K_gpu[0];
  Y = ( depth * ( (float)y - K_gpu[5] ) ) / K_gpu[4];
  Z = depth;

}
//############################################################################
// Explicit instantiations:
template __device__ void  backProjectPixelKernel( const float &depth,
const int &x,
const int &y,
float &X,
float &Y,
float &Z,
const float *K_gpu );



//############################################################################
template< typename T>
__global__ void  backProjectDepthMapKernel( const T *pDepth,
                                            T       *p3D,
                                            int     width,
                                            int     height,
                                            const T       *K_gpu )
{

  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // Do nothing for the pixel-positions which are outside the image
  if ( ( x >= width) || ( y >= height) )
    return;

  // Pixel address
  int     i     = y  * width + x;

  //define 3d points;
  float X, Y, Z;

  backProjectPixelKernel( pDepth[i], x, y, X, Y, Z, K_gpu );

  p3D[i] = X;
  p3D[i + width*height] = Y;
  p3D[i + 2*width*height] = Z;
}
//############################################################################
// Explicit instantiations:
template __global__ void  backProjectDepthMapKernel( const float *pDepth,
float       *p3D,
int         width,
int         height,
const float       *K_gpu );





//############################################################################
template< typename T>
void create3Dpoints( const T * depth,
                     const T * intrinsics_dvc,
                     int       width,
                     int       height,
                     T       * pts_3d)
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  backProjectDepthMapKernel<<< dimGrid, dimBlock >>>(
                                                     depth,
                                                     pts_3d,
                                                     width,
                                                     height,
                                                     intrinsics_dvc);

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

}
//############################################################################
// Explicit instantiations:
template void create3Dpoints( const float * depth,
const float * intrinsics_dvc,
int       width,
int       height,
float       * pts_3d);


//############################################################################
template< typename T>
__global__ void  createNormalsKernel( const T *pDepth,
                                      const float *K_gpu,
                                      T       *pNormals,
                                      int         width,
                                      int         height )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // Do nothing for the pixel-positions which are outside the image
  // Note: Also exclude last column and row, because we cannot compute
  // forward differences there
  if ( ( x >= width) || ( y >= height) )
    return;

  int img_size = width * height;

  // Pixel adress
  size_t     i     = y  * width + x;
  size_t     i1    = i + img_size;
  size_t     i2    = i + 2*img_size;


  //define 3d points
  float aX1, aY1, aZ1, bX1, bY1, bZ1;
  float aX2, aY2, aZ2, bX2, bY2, bZ2;

  float dist_inv = 1.f;

  if (y == height-1)
  {
    backProjectPixelKernel( pDepth[i      ], x, y  , aX1, aY1, aZ1, K_gpu );
    backProjectPixelKernel( pDepth[i-width], x, y-1, aX2, aY2, aZ2, K_gpu );
  }
  else if (y == 0)
  {
    backProjectPixelKernel( pDepth[i+width], x, y+1, aX1, aY1, aZ1, K_gpu );
    backProjectPixelKernel( pDepth[i      ], x, y  , aX2, aY2, aZ2, K_gpu );
  }
  else
  {
    backProjectPixelKernel( pDepth[i+width], x, y+1, aX1, aY1, aZ1, K_gpu );
    backProjectPixelKernel( pDepth[i-width], x, y-1, aX2, aY2, aZ2, K_gpu );

    dist_inv = 0.5f;
  }


  if (x == width-1)
  {
    backProjectPixelKernel( pDepth[i  ], x  , y, bX1, bY1, bZ1, K_gpu );
    backProjectPixelKernel( pDepth[i-1], x-1, y, bX2, bY2, bZ2, K_gpu );
  }
  else if (x == 0)
  {
    backProjectPixelKernel( pDepth[i+1], x+1, y, bX1, bY1, bZ1, K_gpu );
    backProjectPixelKernel( pDepth[i  ], x  , y, bX2, bY2, bZ2, K_gpu );
  }
  else
  {
    backProjectPixelKernel( pDepth[i+1], x+1, y, bX1, bY1, bZ1, K_gpu );
    backProjectPixelKernel( pDepth[i-1], x-1, y, bX2, bY2, bZ2, K_gpu );

    dist_inv = 0.5f;
  }


  if (  (aX1==0 && aY1==0 && aZ1==0) || (aX2==0 && aY2==0 && aZ2==0) ||
        (bX1==0 && bY1==0 && bZ1==0) || (bX2==0 && bY2==0 && bZ2==0) )
  {
    pNormals[i]  = 0.f;
    pNormals[i1] = 0.f;
    pNormals[i2] = 0.f;

    return;
  }


  float a1 = (aX1 - aX2)*dist_inv;  //ABx = Bx - Ax
  float a2 = (aY1 - aY2)*dist_inv;  //ABy = By - Ay
  float a3 = (aZ1 - aZ2)*dist_inv;  //ABz = Bz - Az

  float b1 = (bX1 - bX2)*dist_inv;  //ABx = Bx - Ax
  float b2 = (bY1 - bY2)*dist_inv;  //ABy = By - Ay
  float b3 = (bZ1 - bZ2)*dist_inv;  //ABz = Bz - Az


  // Compute cross product
  float  nx = a2*b3 - a3*b2;
  float  ny = a3*b1 - a1*b3;
  float  nz = a1*b2 - a2*b1;

  // Normalize to |normal|_2 = 1.0
  float norm = sqrtf( nx*nx + ny*ny + nz*nz );
  float inv_norm = norm==0.f ? 0.f : 1.0f/norm;
  float nx_normalized = nx * inv_norm;
  float ny_normalized = ny * inv_norm;
  float nz_normalized = nz * inv_norm;

  pNormals[i]  = nx_normalized;
  pNormals[i1] = ny_normalized;
  pNormals[i2] = nz_normalized;
}
//############################################################################
// Explicit instantiations:
template __global__ void  createNormalsKernel( const float *pDepth,
const float *K_gpu,
float       *pNormals,
int         width,
int         height );




//############################################################################
template< typename T>
void createNormals( const T * depth,
                    const T * intrinsics_dvc,
                    int       width,
                    int       height,
                    T       * normals)
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  createNormalsKernel<<< dimGrid, dimBlock >>>( depth,
                                                intrinsics_dvc,
                                                normals,
                                                width,
                                                height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );


}
//############################################################################
// Explicit instantiations:
template void createNormals( const float * depth,
const float * intrinsics_dvc,
int       width,
int       height,
float   * normals);


//############################################################################
template< typename T>
__global__ void  createNormalsKernel( const T *pDepth,
                                      T       *pNormals,
                                      int         width,
                                      int         height )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // Do nothing for the pixel-positions which are outside the image
  // Note: Also exclude last column and row, because we cannot compute
  // forward differences there
  if ( ( x >= width) || ( y >= height) )
    return;

  int img_size = width * height;

  // Pixel adress
  size_t     i     = y  * width + x;
  size_t     i1    = i + img_size;
  size_t     i2    = i + 2*img_size;

  float nx = 0.f;
  float ny = 0.f;
  float nz = -1.f;

  if (x < width-1)
    nx = pDepth[i+1] - pDepth[i];

  if (y < height-1)
    ny = pDepth[i+width] - pDepth[i];


  // Normalize to |normal|_2 = 1.0
  float norm = sqrtf( nx*nx + ny*ny + nz*nz );
  float inv_norm = norm==0.f ? 0.f : 1.0f/norm;
  float nx_normalized = nx * inv_norm;
  float ny_normalized = ny * inv_norm;
  float nz_normalized = nz * inv_norm;

  pNormals[i]  = nx_normalized;
  pNormals[i1] = ny_normalized;
  pNormals[i2] = nz_normalized;
}
//############################################################################
// Explicit instantiations:
template __global__ void  createNormalsKernel( const float *pDepth,
float       *pNormals,
int         width,
int         height );




//############################################################################
template< typename T>
void createNormals( const T * depth,
                    int       width,
                    int       height,
                    T       * normals)
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  createNormalsKernel<<< dimGrid, dimBlock >>>( depth,
                                                normals,
                                                width,
                                                height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );


}
//############################################################################
// Explicit instantiations:
template void createNormals( const float * depth,
int       width,
int       height,
float   * normals);




//############################################################################
template<typename T>
__global__ void divArray1DKernel( T *num,
                                  const T         *denom,
                                  int             length )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  T val = denom[x];

  if (val < 1e-9)
    num[x] = (T) 0;
  else
    num[x] /= val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray1DKernel<float>(
float           *num,
const float     *denom,
int             length);




//############################################################################
template<typename T>
__global__ void divArray2DKernel( T *num,
                                  const T         *denom,
                                  int             width,
                                  int             height )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int i = x + y * width;

  T val = denom[i];

  if (val < 1e-9)
    num[i] = (T) 0;
  else
    num[i] /= val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray2DKernel<float>(
float           *num,
const float     *denom,
int             width,
int             height);




//############################################################################
template<typename T>
__global__ void divArray3DKernel( T *num,
                                  const T         *denom,
                                  int             width,
                                  int             height,
                                  int             depth )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int i = x + y * width + z * width * height;

  T val = denom[i];
  if (val < 1e-9)
    num[i] = (T) 0;
  else
    num[i] /= val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray3DKernel<float>(
float           *num,
const float     *denom,
int             width,
int             height,
int             depth);


//############################################################################
template<typename T>
void  divArray( T      *num,
                const T * denom,
                int    width,
                int    height,
                int    depth )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  if (height != 1 && depth != 1)//3d case
  {
    get3DGridBlock(width, height, depth, dimGrid, dimBlock);
    divArray3DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height, depth );
  }
  else if (height != 1) //2d case
  {
    get2DGridBlock(width, height, dimGrid, dimBlock);
    divArray2DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height );
  }
  else //1d case
  {
    get1DGridBlock(width, dimGrid, dimBlock);
    divArray1DKernel<<< dimGrid, dimBlock >>>( num, denom, width );
  }

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  divArray<float>( float      *num,
const float * denom,
int    width,
int    height,
int    depth );




//############################################################################
template<typename T>
__global__ void divArray1DKernel( T *num,
                                  const T         denom,
                                  int             length )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  if (denom < 1e-9)
    num[x] = (T) 0;
  else
    num[x] /= denom;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray1DKernel<float>(
float           *num,
const float     denom,
int             length);




//############################################################################
template<typename T>
__global__ void divArray2DKernel( T *num,
                                  const T         denom,
                                  int             width,
                                  int             height )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int i = x + y * width;


  if (denom < 1e-9)
    num[i] = (T) 0;
  else
    num[i] /= denom;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray2DKernel<float>(
float           *num,
const float     denom,
int             width,
int             height);




//############################################################################
template<typename T>
__global__ void divArray3DKernel( T *num,
                                  const T         denom,
                                  int             width,
                                  int             height,
                                  int             depth )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int i = x + y * width + z * width * height;

  if (denom < 1e-9)
    num[i] = (T) 0;
  else
    num[i] /= denom;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray3DKernel<float>(
float           *num,
const float     denom,
int             width,
int             height,
int             depth);


//############################################################################
template<typename T>
void  divArray( T      *num,
                const T  denom,
                int    width,
                int    height,
                int    depth )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  if (height != 1 && depth != 1)//3d case
  {
    get3DGridBlock(width, height, depth, dimGrid, dimBlock);
    divArray3DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height, depth );
  }
  else if (height != 1) //2d case
  {
    get2DGridBlock(width, height, dimGrid, dimBlock);
    divArray2DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height );
  }
  else //1d case
  {
    get1DGridBlock(width, dimGrid, dimBlock);
    divArray1DKernel<<< dimGrid, dimBlock >>>( num, denom, width );
  }

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  divArray<float>( float      *num,
const float  denom,
int    width,
int    height,
int    depth );




//############################################################################
template<typename T>
__global__ void divArray1DKernel( T num,
                                  T         *denom,
                                  int             length )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  T val = denom[x];

  if (val < 1e-9)
    denom[x] = (T) 0;
  else
    denom[x] = num/val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray1DKernel<float>(
float           num,
float     *denom,
int             length);




//############################################################################
template<typename T>
__global__ void divArray2DKernel( T num,
                                  T               *denom,
                                  int             width,
                                  int             height )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int i = x + y * width;

  T val = denom[i];

  if (val < 1e-9)
    denom[i] = (T) 0;
  else
    denom[i] = num/val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray2DKernel<float>(
float           num,
float     *denom,
int             width,
int             height);




//############################################################################
template<typename T>
__global__ void divArray3DKernel( T num,
                                  T               *denom,
                                  int             width,
                                  int             height,
                                  int             depth )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int i = x + y * width + z * width * height;

  T val = denom[i];
  if (val < 1e-9)
    denom[i] = (T) 0;
  else
    denom[i] = num/val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  divArray3DKernel<float>(
float           num,
float     *denom,
int             width,
int             height,
int             depth);


//############################################################################
template<typename T>
void  divArray( T num,
                T * denom,
                int    width,
                int    height,
                int    depth )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  if (height != 1 && depth != 1)//3d case
  {
    get3DGridBlock(width, height, depth, dimGrid, dimBlock);
    divArray3DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height, depth );
  }
  else if (height != 1) //2d case
  {
    get2DGridBlock(width, height, dimGrid, dimBlock);
    divArray2DKernel<<< dimGrid, dimBlock >>>( num, denom, width, height );
  }
  else //1d case
  {
    get1DGridBlock(width, dimGrid, dimBlock);
    divArray1DKernel<<< dimGrid, dimBlock >>>( num, denom, width );
  }

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  divArray<float>( float num,
float * denom,
int    width,
int    height,
int    depth );



//############################################################################
template<typename T>
__global__ void  invertElements1DKernel( T  *arr,
                                         int    length )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= length )
    return;

  // Compute the inverse elements
  if ( arr[x] < 1e-09f )
    arr[x] = (T) 0;
  else
    arr[x] = ((T) 1) / arr[x];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void invertElements1DKernel( float  *arr,
int    length );



//############################################################################
template<typename T>
void invertElements1D( T  *arr,
                       int    length )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  invertElements1DKernel<<< dimGrid, dimBlock >>>( arr, length );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void invertElements1D( float  *arr,
int    length );



//############################################################################
template<typename T>
__global__ void  invertElements2DKernel( T  *arr_2d,
                                         int    width,
                                         int    height)
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int pos = y*width + x;

  T val = arr_2d[pos];

  // Compute the inverse elements
  if ( val < 1e-09f )
    arr_2d[pos] = (T) 0;
  else
    arr_2d[pos] = ((T) 1) / val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void invertElements2DKernel( float  *arr_2d,
int    width,
int    height );



//############################################################################
template<typename T>
void invertElements2D( T  *arr_2d,
                       int    width,
                       int    height )
{
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  invertElements2DKernel<<< dimGrid, dimBlock >>>( arr_2d, width, height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void invertElements2D( float  *arr_2d,
int    width,
int    height );



//############################################################################
template<typename T>
__global__ void  invertElements3DKernel( T  *arr_3d,
                                         int    width,
                                         int    height,
                                         int    depth)
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int pos = z*width*height + y*width + x;

  T val = arr_3d[pos];
  // Compute the inverse elements
  if ( val < 1e-09f )
    arr_3d[pos] = (T) 0;
  else
    arr_3d[pos] = ((T) 1) / val;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void invertElements3DKernel( float  *arr_3d,
int    width,
int    height,
int    depth );



//############################################################################
template<typename T>
void invertElements3D( T  *arr_3d,
                       int    width,
                       int    height,
                       int    depth )
{
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  invertElements3DKernel<<< dimGrid, dimBlock >>>( arr_3d, width, height, depth );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void invertElements3D( float  *arr_3d,
int    width,
int    height,
int    depth );




//############################################################################
template<typename T>
__global__ void mulArray1DKernel( T    *arr,
                                  int  dataSize,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= dataSize )
    return;

  arr[x] *= value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray1DKernel<float>(
float           *arr,
int             dataSize,
float           value );
template  __global__  void  mulArray1DKernel<unsigned char>(
unsigned char   *arr,
int             dataSize,
unsigned char   value );
template  __global__  void  mulArray1DKernel<int>(
int             *arr,
int             dataSize,
int             value );
template  __global__  void  mulArray1DKernel<uint64_t>(
uint64_t        *arr,
int             dataSize,
uint64_t        value );


//############################################################################
template<typename T>
void  mulArray1D( T      *arr,
                  int    length,
                  T      value )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  mulArray1DKernel<<< dimGrid, dimBlock >>>( arr, length, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  mulArray1D<float>( float     *arr,
int       dataSize,
float     value );
template  void  mulArray1D<unsigned char>( unsigned char     *arr,
int               dataSize,
unsigned char     value );
template  void  mulArray1D<int>( int     *arr,
int     dataSize,
int     value );
template  void  mulArray1D<uint64_t>( uint64_t     *arr,
int          dataSize,
uint64_t     value );




//############################################################################
template<typename T>
__global__ void mulArray2DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int i = x + y * width;

  arr[i] *= value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray2DKernel<float>(
float           *arr,
int             width,
int             height,
float           value );
template  __global__  void  mulArray2DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
unsigned char   value );
template  __global__  void  mulArray2DKernel<int>(
int             *arr,
int             width,
int             height,
int             value );
template  __global__  void  mulArray2DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
uint64_t        value );


//############################################################################
template<typename T>
void  mulArray2D( T      *arr,
                  int    width,
                  int    height,
                  T      value )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  mulArray2DKernel<<< dimGrid, dimBlock >>>( arr, width, height, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  mulArray2D<float>( float     *arr,
int               width,
int               height,
float             value );
template  void  mulArray2D<unsigned char>( unsigned char     *arr,
int               width,
int               height,
unsigned char     value );
template  void  mulArray2D<int>( int     *arr,
int               width,
int               height,
int               value );
template  void  mulArray2D<uint64_t>( uint64_t     *arr,
int               width,
int               height,
uint64_t          value );




//############################################################################
template<typename T>
__global__ void mulArray3DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  int  depth,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int i = x + y * width + z * width * height;

  arr[i] *= value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray3DKernel<float>(
float           *arr,
int             width,
int             height,
int  depth,
float           value );
template  __global__  void  mulArray3DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
int  depth,
unsigned char   value );
template  __global__  void  mulArray3DKernel<int>(
int             *arr,
int             width,
int             height,
int  depth,
int             value );
template  __global__  void  mulArray3DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
int  depth,
uint64_t        value );


//############################################################################
template<typename T>
void  mulArray3D( T      *arr,
                  int    width,
                  int    height,
                  int    depth,
                  T      value )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  mulArray3DKernel<<< dimGrid, dimBlock >>>( arr, width, height, depth, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  mulArray3D<float>( float     *arr,
int               width,
int               height,
int    depth,
float             value );
template  void  mulArray3D<unsigned char>( unsigned char     *arr,
int               width,
int               height,
int    depth,
unsigned char     value );
template  void  mulArray3D<int>( int     *arr,
int               width,
int               height,
int    depth,
int               value );
template  void  mulArray3D<uint64_t>( uint64_t     *arr,
int               width,
int               height,
int    depth,
uint64_t          value );




//############################################################################
template<typename T>
__global__ void mulArray1DKernel( T    *arr,
                                  int  dataSize,
                                  const T    *arr2 )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= dataSize )
    return;

  arr[x] *= arr2[x];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray1DKernel<float>(
float           *arr,
int             dataSize,
const float           *arr2 );
template  __global__  void  mulArray1DKernel<unsigned char>(
unsigned char   *arr,
int             dataSize,
const unsigned char   *arr2 );
template  __global__  void  mulArray1DKernel<int>(
int             *arr,
int             dataSize,
const int             *arr2 );
template  __global__  void  mulArray1DKernel<uint64_t>(
uint64_t        *arr,
int             dataSize,
const uint64_t        *arr2 );




//############################################################################
template<typename T>
__global__ void mulArray2DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  const T    *arr2 )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  int i = x + y * width;

  arr[i] *= arr2[i];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray2DKernel<float>(
float           *arr,
int             width,
int             height,
const float           *arr2 );
template  __global__  void  mulArray2DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
const unsigned char   *arr2 );
template  __global__  void  mulArray2DKernel<int>(
int             *arr,
int             width,
int             height,
const int             *arr2 );
template  __global__  void  mulArray2DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
const uint64_t        *arr2 );




//############################################################################
template<typename T>
__global__ void mulArray3DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  int  depth,
                                  const T    *arr2 )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  int i = x + y * width + z * width * height;

  arr[i] *= arr2[i];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  mulArray3DKernel<float>(
float           *arr,
int             width,
int             height,
int  depth,
const float           *arr2 );
template  __global__  void  mulArray3DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
int  depth,
const unsigned char   *arr2 );
template  __global__  void  mulArray3DKernel<int>(
int             *arr,
int             width,
int             height,
int  depth,
const int             *arr2 );
template  __global__  void  mulArray3DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
int  depth,
const uint64_t        *arr2 );


//############################################################################
template<typename T>
void  mulArray( T      *arr,
                const T * arr2,
                int    width,
                int    height,
                int    depth )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  if (height != 1 && depth != 1)//3d case
  {
    get3DGridBlock(width, height, depth, dimGrid, dimBlock);
    mulArray3DKernel<<< dimGrid, dimBlock >>>( arr, width, height, depth, arr2 );
  }
  else if (height != 1) //2d case
  {
    get2DGridBlock(width, height, dimGrid, dimBlock);
    mulArray2DKernel<<< dimGrid, dimBlock >>>( arr, width, height, arr2 );
  }
  else //1d case
  {
    get1DGridBlock(width, dimGrid, dimBlock);
    mulArray1DKernel<<< dimGrid, dimBlock >>>( arr, width, arr2 );
  }

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  mulArray<float>( float      *arr,
const float * arr2,
int    width,
int    height,
int    depth );
template  void  mulArray<unsigned char>( unsigned char      *arr,
const unsigned char * arr2,
int    width,
int    height,
int    depth);
template  void  mulArray<int>( int      *arr,
const int * arr2,
int    width,
int    height,
int    depth);
template  void  mulArray<uint64_t>( uint64_t      *arr,
const uint64_t * arr2,
int    width,
int    height,
int    depth);




//############################################################################
template<typename T>
__global__ void povArray1DKernel( T    *arr,
                                  int  dataSize,
                                  T    exponent )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= dataSize )
    return;

  arr[x] = powf(arr[x], exponent);
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  povArray1DKernel<float>(
float           *arr,
int             dataSize,
float           exponent );
template  __global__  void  povArray1DKernel<unsigned char>(
unsigned char   *arr,
int             dataSize,
unsigned char   exponent );
template  __global__  void  povArray1DKernel<int>(
int             *arr,
int             dataSize,
int             exponent );
template  __global__  void  povArray1DKernel<uint64_t>(
uint64_t        *arr,
int             dataSize,
uint64_t        exponent );


//############################################################################
template<typename T>
void  povArray1D( T      *arr,
                  int    length,
                  T      exponent )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  povArray1DKernel<<< dimGrid, dimBlock >>>( arr, length, exponent );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  povArray1D<float>( float     *arr,
int       dataSize,
float     exponent );
template  void  povArray1D<unsigned char>( unsigned char     *arr,
int               dataSize,
unsigned char     exponent );
template  void  povArray1D<int>( int     *arr,
int     dataSize,
int     exponent );
template  void  povArray1D<uint64_t>( uint64_t     *arr,
int          dataSize,
uint64_t     exponent );




//############################################################################
template<typename T>
__global__ void povArray2DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  T    exponent )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
    return;

  size_t pos = x + y*width;

  arr[pos] = powf(arr[pos], exponent);
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  povArray2DKernel<float>(
float           *arr,
int             width,
int             height,
float           exponent );
template  __global__  void  povArray2DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
unsigned char   exponent );
template  __global__  void  povArray2DKernel<int>(
int             *arr,
int             width,
int             height,
int             exponent );
template  __global__  void  povArray2DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
uint64_t        exponent );


//############################################################################
template<typename T>
void  povArray2D( T      *arr,
                  int    width,
                  int    height,
                  T      exponent )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  povArray2DKernel<<< dimGrid, dimBlock >>>( arr, width, height, exponent );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  povArray2D<float>( float     *arr,
int           width,
int           height,
float         exponent );
template  void  povArray2D<unsigned char>( unsigned char     *arr,
int           width,
int           height,
unsigned char exponent );
template  void  povArray2D<int>( int     *arr,
int           width,
int           height,
int           exponent );
template  void  povArray2D<uint64_t>( uint64_t     *arr,
int           width,
int           height,
uint64_t      exponent );




//############################################################################
template<typename T>
__global__ void povArray3DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  int  depth,
                                  T    exponent )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( x >= width || y >= height || z >= depth )
    return;

  size_t pos = x + y*width + z*width*height;

  arr[pos] = powf(arr[pos], exponent);
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  povArray3DKernel<float>(
float           *arr,
int             width,
int             height,
int             depth,
float           exponent );
template  __global__  void  povArray3DKernel<unsigned char>(
unsigned char   *arr,
int             width,
int             height,
int             depth,
unsigned char   exponent );
template  __global__  void  povArray3DKernel<int>(
int             *arr,
int             width,
int             height,
int             depth,
int             exponent );
template  __global__  void  povArray3DKernel<uint64_t>(
uint64_t        *arr,
int             width,
int             height,
int             depth,
uint64_t        exponent );


//############################################################################
template<typename T>
void  povArray3D( T      *arr,
                  int    width,
                  int    height,
                  int    depth,
                  T      exponent )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  povArray3DKernel<<< dimGrid, dimBlock >>>( arr, width, height, depth, exponent );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  povArray3D<float>( float     *arr,
int           width,
int           height,
int           depth,
float         exponent );
template  void  povArray3D<unsigned char>( unsigned char     *arr,
int           width,
int           height,
int           depth,
unsigned char exponent );
template  void  povArray3D<int>( int     *arr,
int           width,
int           height,
int           depth,
int           exponent );
template  void  povArray3D<uint64_t>( uint64_t     *arr,
int           width,
int           height,
int           depth,
uint64_t      exponent );



//############################################################################
__global__ void   printArray1DKernel( const int  *arr,
                                      int          pos,
                                      int          size )
{
  // Get the 1D-coordinate of the current thread
  const int   i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= size || i!= pos )
    return;

  printf("arr[%d]: %d\n", pos, arr[pos]);
}



//############################################################################
__global__ void   printArray1DKernel( const float  *arr,
                                      int          pos,
                                      int          size )
{
  // Get the 1D-coordinate of the current thread
  const int   i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= size || i!= pos )
    return;

  //    printf("arr[%d]: %f\n", pos, arr[pos]);
  printf("%f", arr[pos]);
}


//############################################################################
template< typename T >
void   printArray1D( const T  *arr,
                     int          pos,
                     int          size )
{
  // Block = 1D array of threads
  dim3  dimBlock( g_CUDA_blockSize1D, 1, 1 );

  // Grid = 1D array of blocks
  // gridSizeX = ceil( dataSize / nBlocksX )
  int   gridSizeX = (size + dimBlock.x-1) / dimBlock.x;
  dim3  dimGrid( gridSizeX, 1, 1 );

  printArray1DKernel<<< dimGrid, dimBlock >>>(
                                              arr, pos, size );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  return;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void  printArray1D( const float  *arr,
int          pos,
int          size );
template void  printArray1D( const int  *arr,
int          pos,
int          size );






//############################################################################
template< typename T >
__global__ void printArray2DKernel( const T   *arr,
                                    int     pos_x,
                                    int     pos_y,
                                    int     width,
                                    int     height)
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;


  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) || (pos_x != x || pos_y != y) )
    return;

  // Get linear address of current coordinate in the image
  int   imgPos    = y * width + x;

  printf("arr[x,y]=arr[%d,%d]=arr[%d]=%f\n", pos_x, pos_y, imgPos, arr[imgPos]);
}





//############################################################################
template< typename T >
void printArray2D( const T   *arr,
                   int     pos_x,
                   int     pos_y,
                   int     width,
                   int     height )
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock( width, height, dimGrid, dimBlock );

  printArray2DKernel<<< dimGrid, dimBlock >>>( arr, pos_x, pos_y, width, height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  return;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void  printArray2D( const float   *arr,
int     pos_x,
int     pos_y,
int     width,
int     height );





//############################################################################
template< typename T >
void printArray2D( const T   *arr,
                   int     pos,
                   int     width,
                   int     height )
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock( width, height, dimGrid, dimBlock );

  //reminder: pos = y * width + x;
  int pos_x =   pos % width;
  int pos_y = ( pos - pos_x ) / width;

  printArray2DKernel<<< dimGrid, dimBlock >>>( arr, pos_x, pos_y, width, height );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  return;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void  printArray2D( const float   *arr,
int     pos,
int     width,
int     height );






//############################################################################
template< typename T >
__global__ void printArray3DKernel( const T   *arr,
                                    int     pos_x,
                                    int     pos_y,
                                    int     pos_z,
                                    int     width,
                                    int     height,
                                    int     depth)
{
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;


  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height)  || (z >= depth) || (pos_x != x || pos_y != y || pos_z != z) )
    return;

  // Get linear address of current coordinate in the image
  int   imgPos    = z * width * height + y * width + x;

  //    printf("arr[x,y,z]=arr[%d,%d,%d]=arr[%d]=%f\n", pos_x, pos_y, pos_z, imgPos, arr[imgPos]);
  printf("%f", arr[imgPos]);
}





//############################################################################
template< typename T >
void printArray3D( const T   *arr,
                   int     pos_x,
                   int     pos_y,
                   int     pos_z,
                   int     width,
                   int     height,
                   int     depth)
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get3DGridBlock( width, height, depth, dimGrid, dimBlock );

  printArray3DKernel<<< dimGrid, dimBlock >>>( arr, pos_x, pos_y, pos_z, width, height, depth );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  return;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void  printArray3D( const float   *arr,
int     pos_x,
int     pos_y,
int     pos_z,
int     width,
int     height,
int     depth );





//############################################################################
template< typename T >
void printArray3D( const T   *arr,
                   int     pos,
                   int     width,
                   int     height,
                   int     depth)
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get3DGridBlock( width, height, depth, dimGrid, dimBlock );

  //reminder: pos = z * width * height + y * width + x;
  int pos_x =   pos % width;
  int pos_y = ( ( pos - pos_x ) % height ) / width;
  int pos_z = ( pos - pos_x - height * pos_y ) / ( width * height );

  printArray3DKernel<<< dimGrid, dimBlock >>>( arr, pos_x, pos_y, pos_z, width, height, depth );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  return;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template void  printArray3D( const float   *arr,
int     pos,
int     width,
int     height,
int     depth);




//############################################################################
template<typename T>
__global__ void setArray1DKernel( T    *arr,
                                  size_t  dataSize,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const size_t   x = blockIdx.x * blockDim.x + threadIdx.x;

  if ( x >= dataSize )
    return;

  arr[x] = value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  setArray1DKernel<float>(
float     *arr,
size_t       dataSize,
float     value );
template  __global__  void  setArray1DKernel<unsigned char>(
unsigned char     *arr,
size_t               dataSize,
unsigned char     value );
template  __global__  void  setArray1DKernel<int>(
int       *arr,
size_t       dataSize,
int       value );
template  __global__  void  setArray1DKernel<uint64_t>(
uint64_t     *arr,
size_t          dataSize,
uint64_t     value );



//############################################################################
template<typename T>
void  setArray1D( T      *data,
                  size_t    length,
                  T      value )
{
  // Block = 1D array of threads
  dim3  dimGrid, dimBlock;
  get1DGridBlock(length, dimGrid, dimBlock);

  setArray1DKernel<<< dimGrid, dimBlock >>>( data, length, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  setArray1D<float>( float     *arr,
size_t       dataSize,
float     value );
template  void  setArray1D<unsigned char>( unsigned char     *arr,
size_t               dataSize,
unsigned char     value );
template  void  setArray1D<int>( int     *arr,
size_t     dataSize,
int     value );
template  void  setArray1D<uint64_t>( uint64_t     *arr,
size_t          dataSize,
uint64_t     value );




//############################################################################
template<typename T>
__global__ void setArray2DKernel( T    *arr,
                                  size_t  width,
                                  size_t  height,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const size_t   x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t   y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( (x >= width) || (y >= height) )
    return;

  // Get linear address of current coordinate in the image
  size_t   i    = y * width + x;

  arr[i] = value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  setArray2DKernel<float>(
float     *arr,
size_t       width,
size_t       height,
float     value );
template  __global__  void  setArray2DKernel<unsigned char>(
unsigned char     *arr,
size_t               width,
size_t               height,
unsigned char     value );
template  __global__  void  setArray2DKernel<int>(
int       *arr,
size_t       width,
size_t       height,
int       value );
template  __global__  void  setArray2DKernel<uint64_t>(
uint64_t     *arr,
size_t          width,
size_t          height,
uint64_t     value );



//############################################################################
template<typename T>
void  setArray2D( T * arr_2d,
                  size_t     width,
                  size_t     height,
                  T       value )
{
  // Block = 2D array of threads
  dim3  dimGrid, dimBlock;
  get2DGridBlock(width, height, dimGrid, dimBlock);

  setArray2DKernel<<< dimGrid, dimBlock >>>( arr_2d, width, height, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  setArray2D<float>( float     *arr_2d,
size_t       width,
size_t       height,
float     value );
template  void  setArray2D<unsigned char>( unsigned char     *arr_2d,
size_t               width,
size_t               height,
unsigned char     value );
template  void  setArray2D<int>( int     *arr_2d,
size_t     width,
size_t     height,
int     value );
template  void  setArray2D<uint64_t>( uint64_t     *arr_2d,
size_t           width,
size_t           height,
uint64_t      value );




//############################################################################
template<typename T>
__global__ void setArray3DKernel( T    *arr,
                                  int  width,
                                  int  height,
                                  int  depth,
                                  T    value )
{
  // Get the 1D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  const int   z = blockIdx.z * blockDim.z + threadIdx.z;

  if ( (x >= width) || (y >= height) || (z >= depth) )
    return;

  // Get linear address of current coordinate in the image
  size_t  i = x + y * width + z * width * height;

  arr[i] = value;
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  __global__  void  setArray3DKernel<float>(
float     *arr,
int       width,
int       height,
int       depth,
float     value );
template  __global__  void  setArray3DKernel<unsigned char>(
unsigned char     *arr,
int               width,
int               height,
int               depth,
unsigned char     value );
template  __global__  void  setArray3DKernel<int>(
int       *arr,
int       width,
int       height,
int       depth,
int       value );
template  __global__  void  setArray3DKernel<uint64_t>(
uint64_t     *arr,
int          width,
int          height,
int          depth,
uint64_t     value );



//############################################################################
template<typename T>
void  setArray3D( T      *data,
                  size_t     width,
                  size_t     height,
                  size_t     depth,
                  T       value )
{
  // Block = 3D array of threads
  dim3  dimBlock, dimGrid;
  get3DGridBlock(width, height, depth, dimGrid, dimBlock);

  setArray3DKernel<<< dimGrid, dimBlock >>>( data, width, height, depth, value );

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template  void  setArray3D<float>( float     *arr,
size_t       width,
size_t       height,
size_t       depth,
float     value );
template  void  setArray3D<unsigned char>( unsigned char     *arr,
size_t               width,
size_t               height,
size_t               depth,
unsigned char     value );
template  void  setArray3D<int>( int     *arr,
size_t     width,
size_t     height,
size_t     depth,
int     value );
template  void  setArray3D<uint64_t>( uint64_t     *arr,
size_t           width,
size_t           height,
size_t           depth,
uint64_t      value );








//############################################################################
// This kernel needs BLOCKSIZE_X*BLOCKSIZE_Y*sizeof(float) bytes of shared
// memory allocated upon kernel invocation, and most important:
//   BLOCKSIZE_X = BLOCKSIZE_Y = 16 !


template<typename T>
__global__ void sum2DKernel16x16( const T    *arr_2d,
                                  int      width,
                                  int      height,
                                  T *   sum)
{
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  const int   pos = threadIdx.x + threadIdx.y * blockDim.x;

  extern volatile __shared__ float sharedMem[];
  const int   iPos = y * width + x;

  if ( (x < width) && (y < height) )
  {
    // Copy data to shared memory
    sharedMem[ pos ] = arr_2d[ iPos ];
  }
  else
  {
    sharedMem[ pos ] = 0;
  }
  __syncthreads();



  if (pos < 128)
  { sharedMem[pos] += sharedMem[pos+128]; }
  __syncthreads();

  if (pos < 64)
  { sharedMem[pos] += sharedMem[pos+64]; }
  __syncthreads();

  if (pos < 32)
  {
    sharedMem[pos] += sharedMem[pos+32];
    sharedMem[pos] += sharedMem[pos+16];
    sharedMem[pos] += sharedMem[pos+8];
    sharedMem[pos] += sharedMem[pos+4];
    sharedMem[pos] += sharedMem[pos+2];
    sharedMem[pos] += sharedMem[pos+1];
  }
  __syncthreads();

  if ( pos == 0 )
    atomicAdd( &(sum[0]), sharedMem[0] );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void sum2DKernel16x16( const float  *arr_2d,
int    width,
int    height,
float *   sum );






//############################################################################
template<typename T>
T   sum2D( const T   *arr_2d,
           int     width,
           int     height )
{
  dim3  dimBlock( 16, 16, 1 );
  int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
  int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
  dim3  dimGrid( gridSizeX, gridSizeY, 1 );

  int sharedMemSizeSum2D = dimBlock.x * dimBlock.y * sizeof(T) + 32;

  T * sum_dvc = NULL;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &sum_dvc, sizeof(T) ) );
  setArray1D(sum_dvc, 1, (T)0);

  sum2DKernel16x16<<< dimGrid, dimBlock, sharedMemSizeSum2D >>>( arr_2d, width, height, sum_dvc );
  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );

  T sum_hst[1];

  CUDA_SAFE_CALL( cudaMemcpy( &(sum_hst[0]), &(sum_dvc[0]), sizeof(T),
      cudaMemcpyDeviceToHost ));

  CUDA_SAFE_CALL(cudaFree(sum_dvc));

  return sum_hst[0];
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template float sum2D( const float  *arr_2d,
int    width,
int    height );








//############################################################################
// This kernel needs BLOCKSIZE_X*BLOCKSIZE_Y*sizeof(float) bytes of shared
// memory allocated upon kernel invocation, and most important:
//   BLOCKSIZE_X = BLOCKSIZE_Y = 16 !


template<typename T>
__global__ void normL22DKernel16x16( const T    *arr_2d,
                                     int      width,
                                     int      height,
                                     T *   norm)
{
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  const int   pos = threadIdx.x + threadIdx.y * blockDim.x;

  extern volatile __shared__ float sharedMem[];
  const int   iPos = y * width + x;

  if ( (x < width) && (y < height) )
  {
    // Copy data to shared memory
    sharedMem[ pos ] = arr_2d[ iPos ];
  }
  else
  {
    sharedMem[ pos ] = 0;
  }
  __syncthreads();



  if (pos < 128)
  { sharedMem[pos] += sharedMem[pos+128]*sharedMem[pos+128]; }
  __syncthreads();

  if (pos < 64)
  { sharedMem[pos] += sharedMem[pos+64]*sharedMem[pos+64]; }
  __syncthreads();

  if (pos < 32)
  {
    sharedMem[pos] += sharedMem[pos+32]*sharedMem[pos+32];
    sharedMem[pos] += sharedMem[pos+16]*sharedMem[pos+16];
    sharedMem[pos] += sharedMem[pos+8]*sharedMem[pos+8];
    sharedMem[pos] += sharedMem[pos+4]*sharedMem[pos+4];
    sharedMem[pos] += sharedMem[pos+2]*sharedMem[pos+2];
    sharedMem[pos] += sharedMem[pos+1]*sharedMem[pos+1];
  }
  __syncthreads();

  if ( pos == 0 )
    atomicAdd( &(norm[0]), sharedMem[0] );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void normL22DKernel16x16( const float  *arr_2d,
int    width,
int    height,
float *   norm );




template<typename T>
__global__ void normL12DKernel16x16( const T    *arr_2d,
                                     int      width,
                                     int      height,
                                     T *   norm)
{
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  const int   pos = threadIdx.x + threadIdx.y * blockDim.x;

  extern volatile __shared__ float sharedMem[];
  const int   iPos = y * width + x;

  if ( (x < width) && (y < height) )
  {
    // Copy data to shared memory
    sharedMem[ pos ] = arr_2d[ iPos ];
  }
  else
  {
    sharedMem[ pos ] = 0;
  }
  __syncthreads();



  if (pos < 128)
  { sharedMem[pos] += sharedMem[pos+128]<0 ? -sharedMem[pos+128] : sharedMem[pos+128]; }
  __syncthreads();

  if (pos < 64)
  { sharedMem[pos] += sharedMem[pos+64]<0 ? -sharedMem[pos+64] : sharedMem[pos+64]; }
  __syncthreads();

  if (pos < 32)
  {
    sharedMem[pos] += sharedMem[pos+32]<0 ? -sharedMem[pos+32] : sharedMem[pos+32];
    sharedMem[pos] += sharedMem[pos+16]<0 ? -sharedMem[pos+16] : sharedMem[pos+16];
    sharedMem[pos] += sharedMem[pos+8]<0 ? -sharedMem[pos+8] : sharedMem[pos+8];
    sharedMem[pos] += sharedMem[pos+4]<0 ? -sharedMem[pos+4] : sharedMem[pos+4];
    sharedMem[pos] += sharedMem[pos+2]<0 ? -sharedMem[pos+2] : sharedMem[pos+2];
    sharedMem[pos] += sharedMem[pos+1]<0 ? -sharedMem[pos+1] : sharedMem[pos+1];
  }
  __syncthreads();

  if ( pos == 0 )
    atomicAdd( &(norm[0]), sharedMem[0] );
}
//----------------------------------------------------------------------------
// Explicit instantiations:
template __global__ void normL12DKernel16x16( const float  *arr_2d,
int    width,
int    height,
float *   norm );






//############################################################################
template<typename T>
T   norm2D( const T   *arr_2d,
            int     width,
            int     height,
            std::string norm)
{
  dim3  dimBlock( 16, 16, 1 );
  int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
  int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
  dim3  dimGrid( gridSizeX, gridSizeY, 1 );

  int sharedMemSizeNorm2D = dimBlock.x * dimBlock.y * sizeof(T) + 32;

  T * norm_dvc = NULL;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &norm_dvc, sizeof(T) ) );
  setArray1D(norm_dvc, 1, (T)0);

  if (norm.compare("L1") == 0)
    normL12DKernel16x16<<< dimGrid, dimBlock, sharedMemSizeNorm2D >>>( arr_2d, width, height, norm_dvc );
  else if (norm.compare("L2") == 0)
    normL22DKernel16x16<<< dimGrid, dimBlock, sharedMemSizeNorm2D >>>( arr_2d, width, height, norm_dvc );
  else
    throw Exception("Error in norm2D(): unknown norm '%s'", norm.c_str());

  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL( cudaPeekAtLastError() );

  T norm_hst[1];

  CUDA_SAFE_CALL( cudaMemcpy( &(norm_hst[0]), &(norm_dvc[0]), sizeof(T),
      cudaMemcpyDeviceToHost ));

  CUDA_SAFE_CALL(cudaFree(norm_dvc));

  return norm.compare("L2")==0 ? std::sqrt(norm_hst[0]) : norm_hst[0];
  }
  //----------------------------------------------------------------------------
  // Explicit instantiations:
  template float norm2D( const float  *arr_2d,
  int    width,
  int    height,
  std::string norm );

}  // namespace CUDA
