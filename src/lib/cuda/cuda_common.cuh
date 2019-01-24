/**
 * \file
 * \brief  The main header file for all CUDA related code
 *
 * \author Georg Kuschk 12/2011
 */

#ifndef MUMFORDSHAS_LIB_CUDA_CUDA_COMMON_CUH_
#define MUMFORDSHAS_LIB_CUDA_CUDA_COMMON_CUH_

//local code
#include "../exception.h"


// NVIDIA SDK - Utilities and system includes
#include <cuda.h>
#include <cuda_runtime_api.h>


// Yes, CUDA declares macros MIN/MAX, but does not check if these are already
// existing!
#ifdef MAX
#undef MAX
#endif
#ifdef MIN
#undef MIN
#endif



#ifdef DEPRECATED
#define CUDA_SAFE_CALL_DEPRECATED( x ) \
    { cudaError_t  error = x; \
    if ( error != cudaSuccess ) \
    throw Exception( "CUDA error: %s\n", cudaGetErrorString(error) ); \
    }
#endif

#define CUDA_SAFE_CALL(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
//  if (code != cudaSuccess)
//  {
//    throw Exception("CUDA_SAFE_CALL Error: %s %s %d\n", cudaGetErrorString(code), file, line);
//  }
}


#define ISNAN_CUDA(x) (isnan(x) || isinf(x))



namespace CUDA
{

  // Gobal variables which every CUDA wrapper can access
  extern bool  g_CUDA_fIsInitialized;

  extern int   g_CUDA_blockSize1D;

  extern int   g_CUDA_blockSize2DX;
  extern int   g_CUDA_blockSize2DY;

  extern int   g_CUDA_blockSize3DX;
  extern int   g_CUDA_blockSize3DY;
  extern int   g_CUDA_blockSize3DZ;

  extern int   g_CUDA_maxSharedMemSize;



  /**
   * \brief  Given width and height of the image, this function returns the
   * 				fastest dimBlock and dimGrid combination for CUDA
   *
   *\params[in]	width			length of the array
   *\params[out]	dimBlock	Output variable describing the 1D array of threads
   *\params[out]	dimGrid		Output variable describing the 1D array of blocks
   */
  void  get1DGridBlock( 	int length,
      dim3 &dimGrid,
      dim3 &dimBlock );



  /**
   * \brief  Given width and height of the image, this function returns the
   * 				fastest dimBlock and dimGrid combination for CUDA
   *
   *\params[in]	width			Width of the image
   *\params[in]	height		Height of the image
   *\params[out]	dimBlock	Output variable describing the 2D array of threads
   *\params[out]	dimGrid		Output variable describing the 2D array of blocks
   */
  void  get2DGridBlock( 	int width,
      int height,
      dim3 &dimGrid,
      dim3 &dimBlock );



  /**
   * \brief  Given width and height of the image, this function returns the
   * 				fastest dimBlock and dimGrid combination for CUDA
   *
   *\params[in]	width			Width of the cube
   *\params[in]	height		Height of the cube
   *\params[in]	depth			Height of the cube
   *\params[out]	dimBlock	Output variable describing the 3D array of threads
   *\params[out]	dimGrid		Output variable describing the 3D array of blocks
   */
  void  get3DGridBlock( 	int width,
      int height,
      int depth,
      dim3 &dimGrid,
      dim3 &dimBlock );




  /**
   * \brief  Initializes the GPU and sets the global parameters according to
   *         the found configuration
   *
   * \param  fVerbose Switches verbose mode on/off
   *
   * \exception Exception In case of an error, an exception is thrown
   */
  void  GPUinit( bool multiple_devices=false );




  /**
   * \brief  Close / reset the GPU after usage at the end of your program
   */
  void  GPUclose( void );




  /**
   * \brief  returns bool is wanted memory size is available
   *
   * \param  wantedMemSize    Wanted memory size
   *
   */
  bool  isGPUMemoryAvailable( size_t     wantedMemSize );




  /**
   * \brief  returns bool is wanted memory size is available
   *
   * \param  wantedMemSize    Wanted memory size
   * \param  memGPUFree       Free GPU memory
   * \param  memGPUTotal      Total GPU memory
   *
   */
  bool  isGPUMemoryAvailable( size_t   wantedMemSize,
      size_t   &memGPUFree,
      size_t   &memGPUTotal );

}//namespace CUDA

#endif  // MUMFORDSHAS_LIB_CUDA_CUDA_COMMON_CUH_

