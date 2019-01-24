/**
 * \file
 * \brief  The main cu file for all CUDA related global variables
 *
 * \author Georg Kuschk 05/2013
 */

//cpp lib
#include <iostream> //cout

//local CUDA
#include "cuda_common.cuh"

//#define VERBOSE

#ifndef ENABLE_CUDA
#define ENABLE_CUDA
#endif

namespace CUDA
{
  // Gobal variables which every CUDA wrapper can access
  // (see 'CUDA_common.h')
  bool  g_CUDA_fIsInitialized = false;

  int   g_CUDA_blockSize1D  = 0;

  int   g_CUDA_blockSize2DX = 0;
  int   g_CUDA_blockSize2DY = 0;

  int   g_CUDA_blockSize3DX = 0;
  int   g_CUDA_blockSize3DY = 0;
  int   g_CUDA_blockSize3DZ = 0;

  int   g_CUDA_maxSharedMemSize = 0;



  //############################################################################
  void  get1DGridBlock(	int length,
      dim3 &dimGrid,
      dim3 &dimBlock )
  {
    // Block = 1D array of threads
    dimBlock = dim3( g_CUDA_blockSize1D, 1, 1 );

    // Grid = 1D array of blocks
    int		gridSizeX = (length  + dimBlock.x-1) / dimBlock.x; // = ceil(width/nBlocksX)

    dimGrid = dim3( gridSizeX, 1, 1 );

  }




  //############################################################################
  void  get2DGridBlock( int width,
      int height,
      dim3 &dimGrid,
      dim3 &dimBlock )
  {
    // Block = 2D array of threads
    dimBlock = dim3( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

    // Grid = 2D array of blocks
    int		gridSizeX = (width  + dimBlock.x-1) / dimBlock.x; // = ceil(width/nBlocksX)
    int		gridSizeY = (height + dimBlock.y-1) / dimBlock.y; // = ceil(height/nBlocksY)

    dimGrid = dim3( gridSizeX, gridSizeY, 1 );

  }





  //############################################################################
  void  get3DGridBlock(	int width,
      int height,
      int depth,
      dim3 &dimGrid,
      dim3 &dimBlock )
  {
    // Block = 3D array of threads
    dimBlock = dim3( g_CUDA_blockSize3DX, g_CUDA_blockSize3DY, g_CUDA_blockSize3DZ );

    // Grid = 3D array of blocks
    int		gridSizeX = (width  + dimBlock.x-1) / dimBlock.x; // = ceil(width/nBlocksX)
    int		gridSizeY = (height + dimBlock.y-1) / dimBlock.y; // = ceil(height/nBlocksY)
    int		gridSizeZ = (depth + dimBlock.z-1) / dimBlock.z; // = ceil(depth/nBlocksZ)

    dimGrid = dim3( gridSizeX, gridSizeY, gridSizeZ );

  }




  //############################################################################
  void  GPUinit( bool multiple_devices )
  {
    // Check whether we already initialized the GPU
    if ( g_CUDA_fIsInitialized )
      return;

    //choose device with most memory, if multiple devices are available otherwise choose device number 0
    int dev_number = 0;

    if (multiple_devices)
    {
      int count;


      CUDA_SAFE_CALL( cudaGetDeviceCount(&count) );

      size_t  max_mem = 0;
      int     best_dvc_number;

      for (int iter_dvc = 0; iter_dvc<count; ++iter_dvc)
      {
        int             devID_temp;
        cudaDeviceProp  props_temp;

        CUDA_SAFE_CALL( cudaSetDevice(iter_dvc) );

        CUDA_SAFE_CALL( cudaGetDevice(&devID_temp) );

        if (iter_dvc != devID_temp)
          throw Exception( "Error initializeGPU(): Could not set device %d to get memory properties\n", iter_dvc);

        CUDA_SAFE_CALL( cudaGetDeviceProperties(&props_temp, devID_temp) );

        if (props_temp.totalGlobalMem > max_mem)
        {
          max_mem         = props_temp.totalGlobalMem;
          best_dvc_number = iter_dvc;
        }
      }

      dev_number = best_dvc_number;

    }

    CUDA_SAFE_CALL( cudaSetDevice(dev_number) );

    int             devID;
    cudaDeviceProp  props;

    // Get information about the GPU
    CUDA_SAFE_CALL( cudaGetDevice(&devID) );
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, devID) );

#ifdef VERBOSE
    printf( "\n---------------------- Initializing GPU ----------------------\n" );
    printf( "Device detected: #%d \"%s\" with Compute %d.%d capability\n",
        devID, props.name,
        props.major, props.minor );
    printf( "totalGlobalMem: %.2f MB\n",     (float)props.totalGlobalMem/1024.0f/1024.0f );
    printf( "sharedMemPerBlock: %.2f KB\n",  (float)props.sharedMemPerBlock/1024.0f );
    printf( "registersPerBlock: %d\n",       props.regsPerBlock );
    printf( "warpSize: %d\n",                props.warpSize );
    printf( "maxThreadsPerBlock: %d\n",      props.maxThreadsPerBlock );
    printf( "maxThreadsDim: %d x %d x %d\n", props.maxThreadsDim[0],
        props.maxThreadsDim[1],
        props.maxThreadsDim[2] );
    printf( "maxGridSize: %d x %d x %d\n",   props.maxGridSize[0],
        props.maxGridSize[1],
        props.maxGridSize[2] );
    printf( "clockRate: %.2f MHz\n",         (float)props.clockRate/1000.0f );
    printf( "multiProcessorCount: %d\n",     props.multiProcessorCount );
    printf( "canMapHostMemory: %d\n",        props.canMapHostMemory );
#endif

    size_t    memGPUFree  = 0;
    size_t    memGPUTotal = 0;
    CUDA_SAFE_CALL( cudaMemGetInfo( &memGPUFree, &memGPUTotal ) );

#ifdef VERBOSE
    printf( "cudaMemGetInfo: Free/Total[MB]: %.1f/%.1f\n",
        (float)memGPUFree/1048576, (float)memGPUTotal/1048576 );
#endif


    // Check minimum requirements needed

    // CUDA version sufficient?
    bool  fOK = true;
    if ( props.major < 2 )
      fOK = false;
    else if ( ( props.major == 2 ) && ( props.minor < 0 ) )
      fOK = false;

    if ( !fOK )
    {
      throw Exception( "Error initializeGPU(): Need CUDA computing "
          "capabilities > 2.0\nDetected version ist %d.%d\n"
          "Disable CUDA-usage for enabling the CPU algorithms\n",
          props.major, props.minor );
    }

    // CUDA allowing mapping host memory?
    if ( !props.canMapHostMemory )
    {
      throw Exception( "Error initializeGPU(): Device can't map host memory\n"
          "Disable CUDA-usage for enabling the CPU algorithms\n" );
    }

    // CUDA allowing 3D threads-blocks?
    if ( props.maxThreadsDim[2] <= 1 )
    {
      throw Exception( "Error initializeGPU(): Device does not allow 3D "
          "thread-blocks:\nmaxThreadsDim: %d x %d x %d\n"
          "Disable CUDA-usage for enabling the CPU algorithms\n",
          props.maxThreadsDim[0], props.maxThreadsDim[1],
          props.maxThreadsDim[2] );
    }

    // CUDA allowing 3D grid-size?
    if ( props.maxGridSize[2] <= 1 )
    {
      throw Exception( "Error initializeGPU(): Device does not allow 3D "
          "grid-size:\nmaxGridSize: %d x %d x %d\n"
          "Disable CUDA-usage for enabling the CPU algorithms\n",
          props.maxGridSize[0], props.maxGridSize[1],
          props.maxGridSize[2] );
    }


    // Determine number of threads per block
    // Note: Do not use the maximum number of threads per block (occupancy)
    if ( props.maxThreadsPerBlock >= 1024 )
    {
      g_CUDA_blockSize1D  = 512;

      //Do not use 32x32, 32x16 is about 25% faster
      g_CUDA_blockSize2DX = 32;
      g_CUDA_blockSize2DY = 16;

      g_CUDA_blockSize3DX = 16;
      g_CUDA_blockSize3DY = 8;
      g_CUDA_blockSize3DZ = 8;
    }
    else if ( props.maxThreadsPerBlock >= 512 )
    {
      g_CUDA_blockSize1D  = 256;

      g_CUDA_blockSize2DX = 32;
      g_CUDA_blockSize2DY = 8;

      g_CUDA_blockSize3DX = 8;
      g_CUDA_blockSize3DY = 8;
      g_CUDA_blockSize3DZ = 8;
    }
    else if ( props.maxThreadsPerBlock >= 256 )
    {
      g_CUDA_blockSize1D  = 256;

      g_CUDA_blockSize2DX = 16;
      g_CUDA_blockSize2DY = 16;

      g_CUDA_blockSize3DX = 8;
      g_CUDA_blockSize3DY = 8;
      g_CUDA_blockSize3DZ = 4;
    }
    else
    {
      int   nx1D = (int)1 < (int)(props.maxThreadsPerBlock) ? (int)(props.maxThreadsPerBlock) : (int)1;

      int   nx2D = (int)1 < (int)(props.maxThreadsPerBlock/8) ? (int)(props.maxThreadsPerBlock/8) : (int)1;
      int   ny2D = props.maxThreadsPerBlock / nx2D;

      int   nx3D = (int)1 < (int)(props.maxThreadsPerBlock/4) ? (int)(props.maxThreadsPerBlock/4) : (int)1;
      int   nLeft = props.maxThreadsPerBlock / nx3D;
      int   ny3D = (int)1 < (int)(nLeft / 4) ? (int)(nLeft / 4) : (int)1;
      int   nz3D = nLeft / ny3D;

      g_CUDA_blockSize1D  = nx1D;

      g_CUDA_blockSize2DX = nx2D;
      g_CUDA_blockSize2DY = ny2D;

      g_CUDA_blockSize3DX = nx3D;
      g_CUDA_blockSize3DY = ny3D;
      g_CUDA_blockSize3DZ = nz3D;
    }

    //Empirical testings:
    //
    //       |  32x16  |  64x8  | 128x4
    //-------------------------------------
    //muc3k  |  1165ms |  903ms | 916ms
    //cones  |  294ms  |  248ms | 252ms

    g_CUDA_maxSharedMemSize = props.sharedMemPerBlock;

#ifdef VERBOSE
    printf( "blockSize1D = %d ; blockSize2D = %dx%d ; blockSize3D = %dx%dx%d\n",
        g_CUDA_blockSize1D, g_CUDA_blockSize2DX, g_CUDA_blockSize2DY,
        g_CUDA_blockSize3DX, g_CUDA_blockSize3DY, g_CUDA_blockSize3DZ );
    printf( "--------------------------------------------------------------\n\n" );
#endif
    g_CUDA_fIsInitialized = true;
    //cudaDeviceReset();    // Not necessary and very costly
  }





  //############################################################################
  void  GPUclose( void )
  {
    cudaDeviceReset();    // Not necessary and very costly
  }



  //############################################################################
  bool  isGPUMemoryAvailable( size_t     wantedMemSize )
  {
    size_t    memGPUFree  = 0;
    size_t    memGPUTotal = 0;

    CUDA_SAFE_CALL( cudaMemGetInfo( &memGPUFree, &memGPUTotal ) );

#ifdef VERBOSE
    printf( "cudaMemGetInfo: Free/Total[MB]: %.1f/%.1f\n",
        (float)memGPUFree/1048576.0f, (float)memGPUTotal/1048576.0f );
#endif

    return ( memGPUFree > wantedMemSize );
  }



  //############################################################################
  bool  isGPUMemoryAvailable( size_t   wantedMemSize,
      size_t   &memGPUFree,
      size_t   &memGPUTotal,
      int      fVerbose )
  {
    CUDA_SAFE_CALL( cudaMemGetInfo( &memGPUFree, &memGPUTotal ) );

#ifdef VERBOSE
    printf( "cudaMemGetInfo: Free/Total[MB]: %.1f/%.1f\n",
        (float)memGPUFree/1048576.0f, (float)memGPUTotal/1048576.0f );
#endif

    return ( memGPUFree > wantedMemSize );
  }

}//namespace CUDA
