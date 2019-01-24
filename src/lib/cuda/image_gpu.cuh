/**
 * \file
 * \brief  Basic class for GPU allocated (and optionally memory aligned) images
 *
 * \author Georg Kuschk 07/2013
 */

#ifndef MUMFORDSHAH_LIB_CUDA_IMAGE_GPU_CUH_
#define MUMFORDSHAH_LIB_CUDA_IMAGE_GPU_CUH_

#include <iostream>

#ifdef USE_OPENCV
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

//local code
#include "../io_data.h"

//local CUDA code
#include "cuda_common.cuh"
#include "cuda_kernels.cuh"

////////////////////////////////////////////////////////////////////////////////
/////global variables
////////////////////////////////////////////////////////////////////////////////
namespace CUDA {



/**
   * \brief   Simple container class for images residing in GPU device memory
   *
   * Note 1:  Memory layout is by default channel-major then row major:
   *             channel1 [ [x1y1], [x2y1],... [xny1] ],
   *             channel1 [ [x1y2], [x2y2],... [xny2] ],
   *             ...
   *             channel1 [ [x1ym], [x2ym],... [xnym] ],
   *             channel2 [ [x1y1], [x2y1],... [xny1] ],
   *             channel2 [ [x1y2], [x2y2],... [xny2] ],
   *             ...
   *             ...
   *             channelk [ [x1ym], [x2ym],... [xnym] ]
   *
   * Reason:  Expansive coordinate projections and interpolations have to be
   *          computed only for one channel (all the others follow by an addition)
   *
   *
   * \author   Bjoern Haefner [06 2017]
   */
template<typename T>
class ImageGPU {
public:
  /**
       * \brief  Default constructor
       */
  ImageGPU() : width_(0), height_(0), channels_(0), data_(NULL) {}


  /**
       * \brief  Constructor
       *
       * \param  width     The image width
       * \param  height    The image height
       * \param  channels  Number of image channels
       */
  ImageGPU(const ImageGPU<T> &other)
    : width_(other.width()), height_(other.height()),
      channels_(other.channels()), data_(NULL) {
    if (other.data_ != NULL) {
      CUDA_SAFE_CALL(cudaMalloc((void**) &data_, Pixels()*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy((void*) data_, other.data_,
                                Pixels()*sizeof(T), cudaMemcpyDeviceToDevice));
    }
  }


  /**
       * \brief  Constructor
       *
       * \param  width     The image width
       * \param  height    The image height
       * \param  channels  Number of image channels
       */
  ImageGPU(int width, int height, int channels = 1)
    : width_(width), height_(height), channels_(channels), data_(NULL) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));
  }


  /**
       * \brief  Constructor
       *
       * \param  width     The image width
       * \param  height    The image height
       * \param  channels  Number of image channels
       */
  ImageGPU(int width, int height, int channels, T val)
    : width_(width), height_(height), channels_(channels), data_(NULL) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));
    if (channels_ == 1) setArray2D(data_, width_, height_, val);
    else setArray3D(data_, width_, height_, channels_, val);
  }


//  /**
//       * \brief  Constructor
//       *
//       * \param  width     The image width
//       * \param  height    The image height
//       * \param  channels  Number of image channels
//       */
//  ImageGPU(int width, int height, int channels,
//           const T *data, bool on_dvc=false)
//    : width_(width), height_(height), channels_(channels), data_(NULL) {
//    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));

//    if (on_dvc) {
//      CUDA_SAFE_CALL(cudaMemcpy(data_, (void*)data,
//                                Pixels()*sizeof(T), cudaMemcpyDeviceToDevice));
//    } else {
//      CUDA_SAFE_CALL(cudaMemcpy(data_, (void*)data,
//                                Pixels()*sizeof(T), cudaMemcpyHostToDevice));
//    }
//  }


#ifdef USE_OPENCV
  /**
       * \brief  Constructor
       *
       * \param  input_depths is an std::vector<cv::Mat> variable
       */
  ImageGPU(const cv::Mat  &input_img)
    : width_(input_img.cols), height_(input_img.rows),
      channels_(input_img.channels()), data_(NULL) {
    //check if data types fit
    size_t data_type = input_img.elemSize1();
    if (sizeof(T) != data_type)
      throw Exception("Data type of cv::Mat does not fit template data type: sizeof(T) = %d =/= %d.\n", sizeof(T), data_type);

    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));

    // re-alignment from OpenCV to ImageGPU needs to be done
    T *dvc_opcv = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dvc_opcv, Pixels()*sizeof(T)));
    CUDA_SAFE_CALL(cudaMemcpy((void*)dvc_opcv, (void*)input_img.data,
                              Pixels()*sizeof(T), cudaMemcpyHostToDevice));

    CUDA::alignDataOpCV2CUDA(data_, width_, height_, channels_,
                             dvc_opcv, input_img.step1(0), input_img.step1(1));

    CUDA_SAFE_CALL(cudaFree(dvc_opcv));
  }
#endif


  /**
       * \brief  Constructor
       *
       * \param  TODO
       */
  ImageGPU(int width, int height, int channels,
           const T *data,
           const std::string aligned = "row-major", bool on_dvc = false)
    : width_(width), height_(height), channels_(channels), data_(NULL) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));

    if (aligned.compare("row-major") == 0) {  // ==0 means they compare equal
      CUDA_SAFE_CALL(cudaMemcpy((void*)data_, (void*)data,
                                Pixels()*sizeof(T),
                                on_dvc ? cudaMemcpyDeviceToDevice :cudaMemcpyHostToDevice));
    } else if (aligned.compare("matlab") == 0 ||
               aligned.compare("column-major") == 0 ) {  // ==0 means they compare equal
      T *dvc_matlab_data = NULL;
      CUDA_SAFE_CALL(cudaMalloc((void**)&dvc_matlab_data, Pixels()*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy((void*)dvc_matlab_data, (void*)data,
                                Pixels()*sizeof(T),
                                on_dvc ? cudaMemcpyDeviceToDevice :cudaMemcpyHostToDevice));
      alignDataMatlab2CUDA(data_, width_, height_, channels_, dvc_matlab_data);
      CUDA_SAFE_CALL(cudaFree(dvc_matlab_data));
    } else {
      std::cerr << "Error in ImageGPU(): Unknown alignment of data: '" <<
                   aligned <<  std::endl;
    }
  }


#ifdef USE_OPENCV
  /**
       * \brief  Constructor
       *
       * \param  input_depths is an std::vector<cv::Mat> variable
       */
  ImageGPU(const std::vector<cv::Mat>  &imgs) : data_(NULL){
    //check if data types and size fit
    for (int ii_img=0; ii_img<imgs.size(); ii_img++) {
      size_t data_type = imgs[ii_img].elemSize1();
      if (sizeof(T) != data_type)
        throw Exception("Data type of cv::Mat does not fit template data type:\n"
                        " sizeof(T) = %d != %d.\n", sizeof(T), data_type);
      if (imgs[0].rows != imgs[ii_img].rows)
        throw Exception("Rows of cv::Mat are not consistent:\n"
                        "input_imgs[0].rows = %d != %d = input_imgs[%d].rows\n",
                        imgs[0].rows, imgs[ii_img].rows, ii_img);
      if (imgs[0].cols != imgs[ii_img].cols)
        throw Exception("Cols of cv::Mat are not consistent:\n"
                        "input_imgs[0].cols = %d != %d = input_imgs[%d].cols\n",
                        imgs[0].cols, imgs[ii_img].cols, ii_img);
      if (imgs[0].channels() != imgs[ii_img].channels())
        throw Exception("Channels of cv::Mat are not consistent:\n"
                        "input_imgs[0].channels() = %d != %d = input_imgs[%d].channels()\n",
                        imgs[0].channels(), imgs[ii_img].channels(), ii_img);
    }

    width_ = imgs[0].cols;
    height_ = imgs[0].rows;
    size_t channel_per_img = imgs[0].channels();
    channels_ = imgs.size() * channel_per_img;
    size_t pixels_per_img = width_*height_*channel_per_img;

    CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));
    T *dvc_opcv = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dvc_opcv, pixels_per_img*sizeof(T)));

    for (size_t ii_img = 0; ii_img < imgs.size(); ii_img++ ) {
      T *dvc_cuda = &data_[ii_img*pixels_per_img];
      T *hst_opcv = (T*)imgs[ii_img].data;

      CUDA_SAFE_CALL(cudaMemcpy((void*)dvc_opcv, (void*)hst_opcv,
                                pixels_per_img*sizeof(T),
                                cudaMemcpyHostToDevice));
      alignDataOpCV2CUDA(dvc_cuda, width_, height_, channel_per_img, dvc_opcv,
                         imgs[ii_img].step1(0), imgs[ii_img].step1(1));
      // check if errors occured and wait until results are ready
      CUDA_SAFE_CALL(cudaPeekAtLastError());
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    CUDA_SAFE_CALL(cudaFree(dvc_opcv));
  }
#endif


  /**
       * \brief += operator here
       *
       */
  ImageGPU& operator+=(const ImageGPU<T>& rhs) {
    //performs the matrix addition out = this + rhs = img + img
    this->CheckDimensionsAndDataType(rhs);
    if (channels_ == 1) addToArray2D( data_, width_, height_, rhs.data_ );
    else addToArray3D( data_, width_, height_, channels_, rhs.data_ );
    return *this; // return the result by reference
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageGPU operator+(ImageGPU<T> lhs, const ImageGPU<T>& rhs) {
    lhs.CheckDimensionsAndDataType(rhs);
    lhs += rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageGPU& operator*=(const T &val) {
    //performs the matrix multiplication out = this + rhs = img + img
    CheckData();
    if (channels_ == 1) mulArray2D( data_, width_, height_, val );
    else mulArray3D( data_, width_, height_, channels_, val );
    return *this; // return the result by reference
  }


  /**
       * \brief * operator here
       *
       */
  friend ImageGPU operator*(ImageGPU<T> lhs, const T& rhs) {
    lhs.CheckData();
    lhs *= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageGPU operator*(const T& lhs, ImageGPU<T> rhs) {
    rhs.CheckData();
    rhs *= lhs; // reuse compound assignment
    return rhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageGPU& operator*=(const ImageGPU<T>& rhs) {
    //performs the matrix multiplication out = this * rhs = img * img
    this->CheckDimensionsAndData(rhs);
    CUDA::mulArray( data_, rhs.data_, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageGPU operator*(ImageGPU<T> lhs, const ImageGPU<T> &rhs) {
    lhs.CheckDimensionsAndData(rhs);
    lhs *= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief /= operator here (pixel-wise)
       *
       */
  ImageGPU& operator/=(const ImageGPU<T>& denom) {
    //performs the matrix dividing out = this / rhs = img / img (pixel-wise)
    this->CheckDimensionsAndData(denom);
    CUDA::divArray(data_, denom.data_, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageGPU operator/(ImageGPU<T> nom, const ImageGPU<T>& denom) {
    nom.CheckDimensionsAndData(denom);
    nom /= denom; // reuse compound assignment
    return nom; // return the result by value (uses move constructor)
  }


  /**
       * \brief /= operator here (pixel-wise)
       *
       */
  ImageGPU& operator/=(const T &denom) {
    //performs the matrix multiplication out = this / rhs = img / img (pixel-wise)
    CheckData();
    CUDA::divArray( data_, denom, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageGPU operator/(const T &nom, ImageGPU<T> denom) {
    denom.CheckData();
    CUDA::divArray(nom, denom.data_,
                   denom.width(), denom.height(), denom.channels());
    return denom; // return the result by value (uses move constructor)
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageGPU operator/(ImageGPU<T> nom, const T &denom) {
    denom.CheckData();
    nom /= denom; // reuse compound assignment
    return nom; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageGPU& operator-=(const ImageGPU<T>& rhs) {
    //performs the matrix subtraction out = this - rhs = img - img
    this->CheckDimensionsAndData(rhs);
    if (channels_ == 1) subFromArray2D( data_, width_, height_, rhs.data_ );
    else subFromArray3D( data_, width_, height_, channels_, rhs.data_ );
    return *this; // return the result by reference
  }


  /**
       * \brief - operator here
       *
       */
  friend ImageGPU operator-(ImageGPU<T> lhs, const ImageGPU<T>& rhs) {
    //performs the matrix subtraction out = lhs - rhs = img - img
    lhs.CheckDimensionsAndData(rhs);
    lhs -= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief Hard copy assignment
       *
       */
  ImageGPU & operator=( const ImageGPU<T> &other ) {
    if(data_ != NULL)
      CUDA_SAFE_CALL(cudaFree(data_));

    data_ = NULL;
    width_ = other.width();
    height_ = other.height();
    channels_ = other.channels();

    if(other.data_ != NULL) {
      CUDA_SAFE_CALL(cudaMalloc((void**)&data_, Pixels()*sizeof(T)));
      CUDA_SAFE_CALL(cudaMemcpy(data_, other.data_,
                                Pixels()*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    return *this;
  }




  /**
       * \brief  release function, same as destructor
       */
  inline void Release() {
    width_ = height_ = channels_ = 0;
    if (data_ != NULL)
      CUDA_SAFE_CALL(cudaFree(data_));
  }




  /**
       * \brief  Destructor
       */
  ~ImageGPU() {
    if (data_ != NULL)
      CUDA_SAFE_CALL(cudaFree(data_));
  }


  size_t width() const {return width_;}
  size_t height() const {return height_;}
  size_t channels() const {return channels_;}
  size_t Pixels() const {return width_*height_*channels_;}

  T *data_;     //!< Row-major array containing the image data_

protected:
  size_t width_;     //!< Width of the image in pixel
  size_t height_;    //!< Height of the image in pixel
  size_t channels_;  //!< Number of channels_ of the image

  void CompareDimensions(const ImageGPU<T> &img) const {
    if (width_ != img.width() ||
        height_ != img.height() ||
        channels_ != img.channels())
      throw Exception("Error: Image sizes do not fit");
  }
  void CheckData() const {
    if(data_ == NULL)
      throw Exception("Error: Image has no data");
  }
  void CheckDimensionsAndData(const ImageGPU<T> &other) const {
    CompareDimensions(other);
    CheckData();
    other.CheckData();
  }
};  // class ImageGPU
} // namespace CUDA



//############################################################################
//############################################################################
// Global functions:
//############################################################################
//############################################################################




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void abs( CUDA::ImageGPU<T> &img )
{
  if (img.channels() == 1)
    CUDA::abs2D( img.data_, img.width(), img.height());
  else
    CUDA::abs3D( img.data_, img.width(), img.height(), img.channels());


}
//############################################################################
// Explicit instantiations:
template void abs( CUDA::ImageGPU<float> &img );




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void add( CUDA::ImageGPU<T> &img, T val )
{
  if (img.channels() == 1)
    CUDA::addToArray2D( img.data_, img.width(), img.height(), val );
  else
    CUDA::addToArray3D( img.data_, img.width(), img.height(), img.channels(), val );

}
//############################################################################
// Explicit instantiations:
template void add( CUDA::ImageGPU<float> &img, float val );



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void copyDeviceToHost(const CUDA::ImageGPU<T> &img_dvc, T *arr_hst,
                      std::string aligned = "row-major",
                      size_t step_y=0, size_t step_x=0) {
  size_t channels = img_dvc.channels();
  size_t width = img_dvc.width();
  size_t height = img_dvc.height();
  size_t pixels = width*height*channels;

  T * arr_aligned_dvc = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**) &arr_aligned_dvc, pixels*sizeof(T)));

  if(aligned.compare("row-major") == 0)
  {
    CUDA_SAFE_CALL(cudaMemcpy((void*)arr_aligned_dvc, (void*)img_dvc.data_, pixels*sizeof(T), cudaMemcpyDeviceToDevice))
  }
  else if (aligned.compare("OpenCV") == 0)
  {
    if (step_x == 0 || step_y == 0) throw Exception("If alignment shall be OpenCV, you need to provide step sizes!");
    CUDA::alignDataCUDA2OpCV( img_dvc.data_, width, height, channels, arr_aligned_dvc, step_y, step_x);
  }
  else if(aligned.compare("matlab") == 0 || aligned.compare("colum-major") == 0)
  {
    CUDA::alignDataCUDA2Matlab( img_dvc.data_, width, height, channels, arr_aligned_dvc);
  }
  else
    throw Exception("Error in copyDeviceToHost(): Unknown alignment %s", aligned.c_str());

  //    *arr_hst = new T[pixels];

  CUDA_SAFE_CALL(cudaMemcpy((void*)(arr_hst), arr_aligned_dvc, pixels*sizeof(T), cudaMemcpyDeviceToHost))

      CUDA_SAFE_CALL(cudaFree(arr_aligned_dvc));

}
//############################################################################
// Explicit instantiations:
template
void copyDeviceToHost( const CUDA::ImageGPU<float> &img_dvc, float *arr_hst, std::string aligned, size_t  step_y, size_t  step_x );



//############################################################################
//############################################################################
//############################################################################
#ifdef USE_OPENCV
template< typename T >
void copyDeviceToHost( const CUDA::ImageGPU<T> &img_dvc, cv::Mat &img_hst ) {
  size_t channels = img_dvc.channels();
  size_t width = img_dvc.width();
  size_t height = img_dvc.height();
  size_t pixels = width*height*channels;

  if(channels >= 4)
    throw Exception("This amount of channels %d, is not yet implemented. Maybe use copyDeviceToHost with std::vector<cv::Mat> as output", channels);

  int data_type = cv::DataDepth<T>::value + (channels-1)*8;
  img_hst.create(height, width, data_type);

  T * p_data_dvc = img_dvc.data_;
  T * p_data_hst = (T*) img_hst.data;

  if (img_hst.channels() == 1) {
    CUDA_SAFE_CALL( cudaMemcpy( (void*)p_data_hst, (void*)p_data_dvc, pixels*sizeof(T), cudaMemcpyDeviceToHost ) );
  } else {
    T *dvc_opcv = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dvc_opcv, pixels*sizeof(T)));

    size_t step_y = img_hst.step1(0);
    size_t step_x = img_hst.step1(1);

    CUDA::alignDataCUDA2OpCV(p_data_dvc,
                             width, height, channels,
                             dvc_opcv,
                             step_y, step_x);
    CUDA_SAFE_CALL(cudaMemcpy((void*)p_data_hst, (void*)dvc_opcv, pixels*sizeof(T), cudaMemcpyDeviceToHost ) );
    CUDA_SAFE_CALL(cudaFree(dvc_opcv));
  }

}
//############################################################################
// Explicit instantiations:
template void copyDeviceToHost( const CUDA::ImageGPU<float> &img_dvc, cv::Mat &img_hst  );
template void copyDeviceToHost( const CUDA::ImageGPU<bool>  &img_dvc, cv::Mat &img_hst  );



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void copyDeviceToHost( const CUDA::ImageGPU<T> &imgs_dvc, std::vector<cv::Mat> &imgs_hst )
{
  size_t channels = imgs_dvc.channels();
  size_t width = imgs_dvc.width();
  size_t height = imgs_dvc.height();
  size_t pixels = width*height;

  int data_type = cv::DataDepth<T>::value;

  for (size_t c = 0; c<channels; c++)
  {
    cv::Mat channel(height, width, data_type);

    CUDA_SAFE_CALL( cudaMemcpy( (void*) channel.data,
                                (void*) (&(imgs_dvc.data_[c*pixels])),
                    pixels*sizeof(T), cudaMemcpyDeviceToHost ) );

    if (channel.channels() > 1)
      throw Exception("Error in copyDeviceToHost(): 1-channel image has surprisingly %d channels", channel.channels());

    imgs_hst.push_back(channel);
  }

}
//############################################################################
// Explicit instantiations:
template
void copyDeviceToHost( const CUDA::ImageGPU<float> &img_dvc, std::vector<cv::Mat> &img_hst  );
#endif



//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 >
void applyMask( CUDA::ImageGPU<T1> &img, const CUDA::ImageGPU<T2> &mask ) {
  int channels = img.channels();
  int width = img.width();
  int height = img.height();

  if (mask.width() != width || mask.height() != height)
    throw Exception("Error in applyMask(): mask should have same size as input ImageGPU<T> img");
  if (mask.channels() != 1)
    throw Exception ("Error in applyMask(): mask should be one channel only. It will be applies per channel to input ImageGPU<T> img.");
  CUDA::apply2DMask( img.data_, width, height, channels, mask.data_ );
}
//############################################################################
// Explicit instantiations:
template void applyMask(CUDA::ImageGPU<float> &img, const CUDA::ImageGPU<float> &mask);
template void applyMask(CUDA::ImageGPU<float> &img, const CUDA::ImageGPU<bool> &mask);



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void getData(const CUDA::ImageGPU<T> &img, T *arr_hst, std::string aligned = "row-major",
             size_t  step_y = 0, size_t  step_x = 0) {
  if(aligned.compare("row-major") == 0) {
    copyDeviceToHost(img, arr_hst);
  } else if (aligned.compare("opencv") == 0) {
    if (step_x == 0 && step_y == 0)
      throw Exception("If alignment shall be OpenCV, you need to provide step sizes!");
    copyDeviceToHost(img, arr_hst, "OpenCV", step_y, step_x);
  } else if(aligned.compare("matlab") == 0 ||
            aligned.compare("colum-major") == 0) {
    copyDeviceToHost(img, arr_hst, "colum-major");
  } else {
    throw Exception("Error in getData(): Unknown alignment %s", aligned.c_str());
  }
}
//############################################################################
// Explicit instantiations:
template
void getData(const CUDA::ImageGPU<float> &img, float *arr, std::string aligned, size_t step_y, size_t step_x);



//############################################################################
//############################################################################
//############################################################################
#ifdef USE_OPENCV
template< typename T >
void image2OpenCV(const CUDA::ImageGPU<T> &img, cv::Mat &mat) {
  size_t channels = img.channels();
  size_t width = img.width();
  size_t height = img.height();

  if(channels >= 4)
    throw Exception("This amount of channels_ %d, is not yet implemented. Maybe use copyDeviceToHost with std::vector<cv::Mat> as output", channels);

  int data_type = cv::DataDepth<T>::value + (channels-1)*8;
  mat.create(height, width, data_type);
  T *p_mat = (T*) mat.data;
  getData(img, p_mat, "opencv", mat.step1(0), mat.step1(1));
}
//############################################################################
// Explicit instantiations:
template void image2OpenCV( const CUDA::ImageGPU<float> &img, cv::Mat &mat );
template void image2OpenCV( const CUDA::ImageGPU<bool>  &img, cv::Mat &mat );
#endif



//############################################################################
//############################################################################
//############################################################################
template<typename T>
void get3Dpoints(const CUDA::ImageGPU<T> &depth,
                 const T *intrinsics_hst,
                 CUDA::ImageGPU<T> &pts_3d) {
  T* intrinsics_dvc = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&intrinsics_dvc, 6*sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy((void*)intrinsics_dvc, (void*) intrinsics_hst, 6*sizeof(T), cudaMemcpyHostToDevice));

  pts_3d  = CUDA::ImageGPU<float>( depth.width(), depth.height(), 3, 0.f );

  CUDA::create3Dpoints( depth.data_, intrinsics_dvc, depth.width(), depth.height(), pts_3d.data_ );

  CUDA_SAFE_CALL(cudaFree(intrinsics_dvc));

}
//############################################################################
// Explicit instantiations:
template void get3Dpoints(const CUDA::ImageGPU<float> &depth,
const float *intrinsics_hst,
CUDA::ImageGPU<float> &pts_3d);




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void getChannel(const CUDA::ImageGPU<T> &img,
                size_t channel, CUDA::ImageGPU<T> &chn) {
  if (channel >= img.channels())
    return;
  size_t width = img.width();
  size_t height = img.height();
  const size_t    img_size = width * height;

  chn.Release();
  chn = CUDA::ImageGPU<T>(width, height, 1, &(img.data_[channel*img_size]),
      "row-major", true);

  return;
}
//############################################################################
// Explicit instantiations:
template
void getChannel( const CUDA::ImageGPU<float>      &img,
size_t          channel,
CUDA::ImageGPU<float> &chn  );




//############################################################################
//############################################################################
//############################################################################
template< typename T>
void getNormals( const CUDA::ImageGPU<T> &depth,
                 const T *         intrinsics_hst,
                 CUDA::ImageGPU<T>       &normals)
{
  T* intrinsics_dvc = NULL;
  CUDA_SAFE_CALL(cudaMalloc((void**)&intrinsics_dvc, 6*sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy((void*)intrinsics_dvc, (void*) intrinsics_hst, 6*sizeof(T), cudaMemcpyHostToDevice));

  normals = CUDA::ImageGPU<float>( depth.width(), depth.height(), 3, 0.f );

  createNormals( depth.data_, intrinsics_dvc, depth.width(), depth.height(), normals.data_);

  CUDA_SAFE_CALL(cudaFree(intrinsics_dvc));

}




//############################################################################
//############################################################################
//############################################################################
template< typename T>
void getNormals( const CUDA::ImageGPU<T> &depth,
                 CUDA::ImageGPU<T>       &normals)
{
  normals = CUDA::ImageGPU<float>( depth.width(), depth.height(), 3, 0.f );

  createNormals( depth.data_, depth.width(), depth.height(), normals.data_);

}


//############################################################################
//############################################################################
//############################################################################
template< typename T >
void inv( CUDA::ImageGPU<T> &img )
{
  if (img.channels() == 1)
    CUDA::invertElements2D( img.data_, img.width(), img.height() );
  else
    CUDA::invertElements3D( img.data_, img.width(), img.height(), img.channels() );
}
//############################################################################
// Explicit instantiations:
template void inv( CUDA::ImageGPU<float> &img );


//############################################################################
//############################################################################
//############################################################################
template< typename T >
void mul( CUDA::ImageGPU<T> &img, T val )
{
  if (img.channels() == 1)
    CUDA::mulArray2D( img.data_, img.width(), img.height(), val );
  else
    CUDA::mulArray3D( img.data_, img.width(), img.height(), img.channels(), val );
}
//############################################################################
// Explicit instantiations:
template void mul( CUDA::ImageGPU<float> &img, float val );


//############################################################################
//############################################################################
//############################################################################
template< typename T >
float norm( const CUDA::ImageGPU<T> &img, std::string norm = "L1" )
{
  return CUDA::norm2D( img.data_, img.width(), img.height()*img.channels(), norm );
}
//############################################################################
// Explicit instantiations:
template float norm( const CUDA::ImageGPU<float> &img, std::string norm );


//############################################################################
//############################################################################
//############################################################################
template< typename T >
void pov( CUDA::ImageGPU<T> &img, T exponent )
{
  if (exponent == (T) 1) return;

  if (img.channels() == 1)
    CUDA::povArray2D( img.data_, img.width(), img.height(), exponent );
  else
    CUDA::povArray3D( img.data_, img.width(), img.height(), img.channels(), exponent );
}
//############################################################################
// Explicit instantiations:
template void pov( CUDA::ImageGPU<float> &img, float exponent );




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void  saveImage( const std::string &filename, const CUDA::ImageGPU<T> &img )
{
  if ( img.data_ == NULL )
    throw Exception( "Error saveImage(): no data in img\n" );

  size_t width = img.width();
  size_t height = img.height();
  size_t img_size = width*height;
  size_t channels = img.channels();
  size_t pixels = img_size*channels;

  T * data_hst = new T[pixels];

  CUDA_SAFE_CALL( cudaMemcpy((void*)data_hst,(void*)img.data_, pixels*sizeof(T), cudaMemcpyDeviceToHost) );
  StoreArrayAsBinary(filename, data_hst, width, height, channels);

  delete[] data_hst;
}
//############################################################################
// Explicit instantiations:
template
void saveImage( const std::string &filename, const CUDA::ImageGPU<float> &img  );



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void  saveImageChannel( const std::string &filename, const CUDA::ImageGPU<T> &img, size_t channel )
{
  if ( img.data_ == NULL )
    throw Exception( "Error saveImage(): no data in img\n" );

  size_t width = img.width();
  size_t height = img.height();
  size_t img_size = width*height;

  T * data_hst = new T[img_size];

  CUDA_SAFE_CALL( cudaMemcpy((void*)data_hst,(void*)(&(img.data_[channel*img_size])), img_size*sizeof(T), cudaMemcpyDeviceToHost) );
  StoreArrayAsBinary(filename, data_hst, width, height, 1);

  delete[] data_hst;
}
//############################################################################
// Explicit instantiations:
template
void saveImageChannel( const std::string &filename, const CUDA::ImageGPU<float> &img, size_t channel  );



//############################################################################
//############################################################################
//############################################################################
#ifdef USE_OPENCV
template< typename T1, typename T2 = T1 >
void  imshow( const std::string &window_name, const CUDA::ImageGPU<T1> &img_dvc, const T2 &factor = (T2) 1  )
{
  cv::Mat img_hst;
  copyDeviceToHost(img_dvc, img_hst);
  cv::imshow(window_name, factor * img_hst);
}
//############################################################################
// Explicit instantiations:
template void imshow( const std::string &window_name, const CUDA::ImageGPU<float> &img_dvc, const float &factor );
template void imshow( const std::string &window_name, const CUDA::ImageGPU<bool> &img_dv, const float &factor );




//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 = T1 >
void  imwrite( const std::string &file_name, const CUDA::ImageGPU<T1> &img_dvc, const T2 &factor = (T2) 255  )
{
  cv::Mat img_hst;
  copyDeviceToHost(img_dvc, img_hst);
  cv::imwrite(file_name, factor * img_hst);

}
//############################################################################
// Explicit instantiations:
template void imwrite( const std::string &file_name, const CUDA::ImageGPU<float> &img_dvc, const float &factor );
template void imwrite( const std::string &file_name, const CUDA::ImageGPU<bool> &img_dv, const float &factor );
#endif


#endif  // MUMFORDSHAH_LIB_CUDA_IMAGE_GPU_CUH_
