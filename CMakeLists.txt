CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mumfordShah) #project name

######################################################
##################ADJUST##############################
######################################################
#what shall be build?
SET(MEXF FALSE) #generate mex file for matlab usage
SET(EXEF TRUE) #generate executable to call code from commandline
SET(CUDA TRUE) #use cuda for speedup if applicable
SET(DEBUG FALSE) #to debug gpu code in NSight

##################CPP & CUDA FILES####################
######################################################
#file(s) for binary version only
SET(CPP_MAIN_FILES  src/mumford_shah_test.cc
                    )

#file(s) for mex ( e.g. mex entrance function) version only
SET(MEX_MAIN_FILES src/mex/main_mex.cc
                   src/mex/mex_utils.h
                   )

#files needed for both, cpp and cuda built
SET(COMMON_SRC_FILES  src/lib/mumford_shah.cc
                      src/lib/exception.h
                      src/lib/io_data.cc
                      src/lib/image.h
                      )

#cuda files (e.g. kernels)
SET(CU_SRC_FILES  src/lib/cuda/cuda_common.cu src/lib/cuda/cuda_common.cuh
                  src/lib/cuda/cuda_kernels.cu src/lib/cuda/cuda_kernels.cuh
                  src/lib/cuda/mumford_shah_kernels.cu
                  src/lib/cuda/image_gpu.cuh
                  )
#cpp files needed for built with non-cuda support only
SET(CPP_SRC_FILES  src/lib/image_cpu.h
                   src/lib/image_cpu_utils.cc
                   )

##################LIBRARIES###########################
######################################################
#used for cuda build only (not needed for this framework)
SET(CUDA_LIBS )# e.g. CUDA_CUBLAS_LIBRARIES CUDA_CUFFT_LIBRARIES

######################################################
##################SET EVERYTHING UP###################
######################################################
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) #output for executable i.e., EXEF=TRUE
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)    #output for mex-file i.e., MEXF=TRUE

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) #add modules e.g., FindMatlab.cmake

MESSAGE(STATUS "MEXF=" ${MEXF})
MESSAGE(STATUS "EXEF=" ${EXEF})
MESSAGE(STATUS "CUDA=" ${CUDA})
MESSAGE(STATUS "DEBUG=" ${DEBUG})


##################FIND OTHER LIBRARIES################
######################################################
if(${EXEF})
SET(MIN_OPENCV_VERSION )#2.4.9
  FIND_PACKAGE(OpenCV ${MIN_OPENCV_VERSION} REQUIRED)
  ADD_DEFINITIONS(-DUSE_OPENCV)
  ADD_DEFINITIONS(-DAPP_SOURCE_DIR=${PROJECT_SOURCE_DIR})
  if(DEBUG)
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  else(DEBUG)
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
  endif(DEBUG)
endif(${EXEF})
##################FIND MATLAB#########################
######################################################
if(${MEXF})
  FIND_PACKAGE(Matlab REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if matlab is not used
  ADD_DEFINITIONS(-DUSE_MEX)

  # set up matlab libraries
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

  #use correct suffix depending on OS
  if(WIN32) # 32-bit or 64-bit mex
      if (CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw64 )
      else(CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw32 )
      endif(CMAKE_CL_64)
  else(WIN32)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexa64 )
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexglx )
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(WIN32)
endif(${MEXF})

##################FIND CUDA###########################
######################################################
if(${CUDA})
  SET(MIN_CUDA_VERSION 8.0)
  FIND_PACKAGE(CUDA ${MIN_CUDA_VERSION} REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if CUDA is not used
  ADD_DEFINITIONS(-DUSE_CUDA)

  # set SRC_FILES using the common cpp files and cu source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CU_SRC_FILES})

  #to correctly include the above cuda libraries, a splitting must take place
  foreach(libname ${CUDA_LIBS})
    LIST(APPEND CUDA_LIBS_SEP ${${libname}})
  endforeach(${CUDA_LIBS_SEP})

  #compiler flags for cuda
  SET(CUDA_NVCC_FLAGS
  -Xcompiler #Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix. Allowed values for this option: c, c++, cu.
  -fPIC
  -use_fast_math
  #-gencode arch=compute_20,code="sm_20,compute_20"
  #-gencode=arch=compute_30,code="sm_30,compute_30"
  #-gencode=arch=compute_50,code="sm_50,compute_50"
  #-gencode=arch=compute_52,code="sm_52,compute_52"
  #-gencode=arch=compute_60,code="sm_60,compute_60"
  #-gencode=arch=compute_61,code="sm_61,compute_61"
  #-gencode=arch=compute_62,code="sm_62,compute_62"
  # --ptxas-options=-v
  -O3 #specify optimization level for host code
  -Wno-deprecated-gpu-targets #to suppress warning for deprecated architectures
  -std=c++11 #specify c++11 language
  )
  if($DEBUG)
    SET(CUDA_NVCC_FLAGS
    --device-debug #possibility to debug device code; DEBUG==FALSE for speedup
    --debug #possibility to debug host code; DEBUG==FALSE for speedup
    )
  endif($DEBUG)

else(${CUDA})#if cuda is not used 
  #set SRC_FILES using the common cpp files and cpp source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CPP_SRC_FILES} )

endif(${CUDA})

######################################################
##################BUILD FILES#########################
######################################################

######################MEX#############################
######################################################
if(${MEXF})

  #if mex lib is build, add suffix MEX to lib file
  SET(MEX_LIB_NAME "${PROJECT_NAME}MEX")
  
  if(${CUDA})
    CUDA_COMPILE(${MEX_LIB_NAME} ${MEX_MAIN_FILES} ${SRC_FILES} SHARED)   
  endif(${CUDA})

  ADD_LIBRARY(${MEX_LIB_NAME} SHARED ${${MEX_LIB_NAME}} ${MEX_MAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)

  TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                        ${MATLAB_LIBRARIES}
                        )
  if(${CUDA})
    MESSAGE(STATUS "LINKING CUDA LIBRARIES: " ${CUDA_LIBS_SEP})
    MESSAGE(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          )
  endif(${CUDA})

  SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES
                        PREFIX ""
                        SUFFIX ${MEX_SUFFIX}
                        COMPILE_FLAGS "-std=c++11"
                        )
  if(${CUDA})
    SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif(${CUDA})

endif(${MEXF})


##################EXECUTABLE##########################
######################################################
if(${EXEF}) #now take care of the executable

  if(${CUDA}) #if cuda is used
    CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES} )
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          ${OpenCV_LIBS} #load opencv libs
                          )
  else(${CUDA})#is cuda is not used
    ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                          ${OpenCV_LIBS} #load opencv libs
                          )
endif(${CUDA})

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES
                        COMPILE_FLAGS "-std=c++11"
                        )

endif(${EXEF})



