#ifndef MUMFORDSHAH_LIB_IMAGE_CPU_H_
#define MUMFORDSHAH_LIB_IMAGE_CPU_H_

#define HERE printf("here\n");

#include <iostream>
#include <cstring>
#include <math.h>

#ifdef USE_OPENCV
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

//local code
#include "io_data.h"
#include "image_cpu_utils.h"
#include "exception.h"


/**
   * \brief   Simple container class for images residing in CPU device memory
   *
   * Note 1:  Memory layout is by default channel-major then row-major:
   *             channel1 [ [x1,y1], [x2,y1],... [xn,y1] ],
   *             channel1 [ [x1,y2], [x2,y2],... [xn,y2] ],
   *             ...
   *             channel1 [ [x1,ym], [x2,ym],... [xn,ym] ],
   *             channel2 [ [x1,y1], [x2,y1],... [xn,y1] ],
   *             channel2 [ [x1,y2], [x2,y2],... [xn,y2] ],
   *             ...
   *             ...
   *             channelk [ [x1,ym], [x2,ym],... [xn,ym] ]
   *
   * OR
   *
   *             channel1 [ [col1,row1], [col2,row1],... [coln,row1] ],
   *             channel1 [ [col1,row2], [col2,row2],... [coln,row2] ],
   *             ...
   *             channel1 [ [col1,rowm], [col2,rowm],... [coln,rowm] ],
   *             channel2 [ [col1,row1], [col2,row1],... [coln,row1] ],
   *             channel2 [ [col1,row2], [col2,row2],... [coln,row2] ],
   *             ...
   *             ...
   *             channelk [ [col1,rowm], [col2,rowm],... [coln,rowm] ]
   *
   * Reason:  Expansive coordinate projections and interpolations have to be
   *          computed only for one channel (all the others follow by an addition)
   *
   *
   * \author   Bjoern Haefner [06 2017]
   */
template<typename T>
class ImageCPU {
public:
  /**
       * \brief  Default constructor
       */
  ImageCPU() : width_(0), height_(0), channels_(0), data_(NULL) {}


  /**
       * \brief  Constructor
       *
       * \param  width_     The image width_
       * \param  height_    The image height_
       * \param  channels_  Number of image channels_
       */


  ImageCPU(const ImageCPU<T> &other)
    : width_(other.width()), height_(other.height()),
      channels_(other.channels()), data_(NULL) {
    if (other.data_ != NULL) {
      data_ = new T[Pixels()];
      std::memcpy(data_, other.data_, sizeof(T)*Pixels());
    }
  }


  /**
       * \brief  Constructor
       *
       * \param  width_     The image width_
       * \param  height_    The image height_
       * \param  channels_  Number of image channels_
       */


  ImageCPU(int width, int height, int channels = 1)
    : width_(width), height_(height), channels_(channels), data_(NULL) {
    data_ = new T[width_*height_*channels_];
  }


  /**
       * \brief  Constructor
       *
       * \param  width_     The image width_
       * \param  height_    The image height_
       * \param  channels_  Number of image channels_
       */
  ImageCPU(int width, int height, int channels, T val)
    : width_(width), height_(height), channels_(channels), data_(NULL) {
    data_ = new T[Pixels()];
    for (size_t pos=0; pos<Pixels(); pos++)
      data_[pos] = val;
  }


  //  /**
  //       * \brief  Constructor
  //       *
  //       * \param  width_     The image width_
  //       * \param  height_    The image height_
  //       * \param  channels_  Number of image channels_
  //       */
  //  ImageCPU(int width, int height, int channels, const T *data)
  //    : width_(width), height_(height), channels_(channels), data_(NULL) {
  //    if (data != NULL) {
  //      size_t pixels = (size_t)(width_ * height_ * channels_);
  //      data_ = new T[pixels];
  //      std::memcpy(data_, data, sizeof(T)*pixels);
  //    }
  //  }



#ifdef USE_OPENCV
  /**
       * \brief  Constructor
       *
       * \param  input_depths is an std::vector<cv::Mat> variable
       */
  ImageCPU(const cv::Mat  &other_mat)
    : width_(other_mat.cols),
      height_(other_mat.rows),
      channels_(other_mat.channels()),
      data_(NULL) {
    //check if data_ types fit
    size_t data_type = other_mat.elemSize1();
    if (sizeof(T) != data_type)
      throw Exception("Data type of cv::Mat does not fit template data_ type: sizeof(T) = %d =/= %d.\n", sizeof(T), data_type);

    data_ = new T[Pixels()];

    // re-alignment from OpenCV to ImageCPU needs to be done
    T* opcv = (T*)other_mat.data;

    alignDataOpCV2CPU(data_, width_, height_, channels_,
                      opcv, other_mat.step1(0), other_mat.step1(1));

  }
#endif


  /**
       * \brief  Constructor
       *
       * \param  TODO
       */
  ImageCPU(int width, int height, int channels, const T *data,
           const std::string aligned = "row-major",
           size_t step_x = 0, size_t step_y = 0)
    :width_(width), height_(height), channels_(channels), data_(NULL) {
    data_ = new T[Pixels()];

    if (aligned.compare("row-major") == 0) {// ==0 means they compare equal
      std::memcpy(data_, data, sizeof(T)*Pixels());
    } else if (aligned.compare("matlab") == 0 ||
               aligned.compare("column-major") == 0) {// ==0 means they compare equal
      alignDataMatlab2CPU( data_, width_, height_, channels_, data);
    } else if (aligned.compare("opencv") == 0) {// ==0 means they compare equal
      alignDataOpCV2CPU(data_, width_, height_, channels_,
                        data, step_x, step_y);
    } else {
      std::cerr << "Error in ImageCPU(): Unknown alignment of data_: '" << aligned <<  std::endl;
    }
  }


  /**
       * \brief += operator here
       *
       */
  ImageCPU& operator+=(const ImageCPU<T>& rhs) {
    //performs the matrix addition out = this + rhs = img + img
    this->CheckDimensionsAndData(rhs);
    addToArray( data_, width_, height_, channels_, rhs.data_ );
    return *this; // return the result by reference
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageCPU operator+(ImageCPU<T> lhs, const ImageCPU<T>& rhs) {
    lhs.CheckDimensionsAndData(rhs);
    lhs += rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageCPU& operator*=(const T &val) {
    //performs the matrix multiplication out = this + val = img + scalar
    CheckData();
    mulArray(data_, width_, height_, channels_, val);
    return *this; // return the result by reference
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageCPU operator*(ImageCPU<T> lhs, const T& rhs) {
    lhs.CheckData();
    lhs *= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageCPU operator*(const T& lhs, ImageCPU<T> rhs) {
    rhs.CheckData();
    rhs *= lhs; // reuse compound assignment
    return rhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageCPU& operator*=(const ImageCPU<T>& rhs) {
    //performs the matrix multiplication out = this + rhs = img + img
    this->CheckDimensionsAndData(rhs);
    mulArray( data_, rhs.data_, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief + operator here
       *
       */
  friend ImageCPU operator*(ImageCPU<T> lhs, const ImageCPU<T> &rhs) {
    lhs.CheckDimensionsAndData(rhs);
    lhs *= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief /= operator here (pixel-wise)
       *
       */
  ImageCPU& operator/=(const ImageCPU<T>& denom) {
    //performs the matrix multiplication out = this / rhs = img / img pixel-wise
    this->CheckDimensionsAndData(denom);
    divArray( data_, denom.data_, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageCPU operator/(ImageCPU<T> nom, const ImageCPU<T>& denom) {
    nom.CheckDimensionsAndData(denom);
    nom /= denom; // reuse compound assignment
    return nom; // return the result by value (uses move constructor)
  }


  /**
       * \brief /= operator here (pixel-wise)
       *
       */
  ImageCPU& operator/=(const T &denom) {
    //performs the matrix multiplication out = this / rhs = img / img pixel-wise
    CheckData();
    divArray(data_, denom, width_, height_, channels_);
    return *this; // return the result by reference
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageCPU operator/(const T &nom, ImageCPU<T> denom) {
    denom.CheckData();
    divArray(nom, denom.data_, denom.width_, denom.height_, denom.channels_); // reuse compound assignment
    return denom; // return the result by value (uses move constructor)
  }


  /**
       * \brief / operator here (pixel-wise)
       *
       */
  friend ImageCPU operator/(ImageCPU<T> nom, const T &denom) {
    nom.CheckData();
    nom /= denom; // reuse compound assignment
    return nom; // return the result by value (uses move constructor)
  }


  /**
       * \brief += operator here
       *
       */
  ImageCPU& operator-=(const ImageCPU<T>& rhs) {
    //performs the matrix subtraction out = this - rhs = img - img
    this->CheckDimensionsAndData(rhs);
    subFromArray( data_, width_, height_, channels_, rhs.data_ );
    return *this; // return the result by reference
  }


  /**
       * \brief - operator here
       *
       */
  friend ImageCPU operator-(ImageCPU<T> lhs, const ImageCPU<T>& rhs) {
    //performs the matrix subtraction out = lhs - rhs = img - img
    lhs.CheckDimensionsAndData(rhs);
    lhs -= rhs; // reuse compound assignment
    return lhs; // return the result by value (uses move constructor)
  }


  /**
       * \brief Hard copy assignment
       *
       */
  ImageCPU & operator=( const ImageCPU<T> &other ) {
    if(data_ != NULL)
      delete []data_;

    data_ = NULL;
    width_ = other.width();
    height_ = other.height();
    channels_ = other.channels();

    if(other.data_ != NULL) {
      data_ = new T[Pixels()];
      std::memcpy(data_, other.data_, sizeof(T)*Pixels());
    }
    return *this;
  }


  /**
       * \brief  release function, same as destructor
       */
  inline void Release() {
    width_ = height_ = channels_ = 0;
    if ( data_ != NULL ) delete []data_;
    data_ = NULL;
  }




  /**
       * \brief  Destructor
       */
  ~ImageCPU()
  {
    if ( data_ != NULL ) delete []data_;
  }

  size_t width() const {return width_;}
  size_t height() const {return height_;}
  size_t channels() const {return channels_;}
  size_t Pixels() const {return width_*height_*channels_;}

  T     *data_;     //!< Row-major array containing the image data_

protected:
  size_t   width_;     //!< Width of the image in pixel
  size_t   height_;    //!< Height of the image in pixel
  size_t   channels_;  //!< Number of channels_ of the image

  void CompareDimensions(const ImageCPU<T> &img) const {
    if (width_ != img.width() ||
        height_ != img.height() ||
        channels_ != img.channels())
      throw Exception("Error: Image sizes do not fit");
  }
  void CheckData() const {
    if(data_ == NULL)
      throw Exception("Error: Image has no data");
  }
  void CheckDimensionsAndData(const ImageCPU<T> &other) const {
    CompareDimensions(other);
    CheckData();
    other.CheckData();
  }
};  // class ImageCPU




//############################################################################
//############################################################################
// Global functions:
//############################################################################
//############################################################################




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void abs( ImageCPU<T> &img ) {
  for (size_t ch=0; ch<img.channels(); ch++)
    for (size_t y=0; y<img.height(); y++)
      for (size_t x=0; x<img.width(); x++)
      {
        size_t idx = getLIdx(x,y,ch,img.width(),img.height());
        T val = img.data_[idx];
        img.data_[idx] = (val < ((T) 0)) ? -val : val;
      }
}
//############################################################################
// Explicit instantiations:
template void abs( ImageCPU<float> &img );




//############################################################################
//############################################################################
//############################################################################
template< typename T >
ImageCPU<T> abs(ImageCPU<T> img) {
  for (size_t ch=0; ch<img.channels(); ch++)
    for (size_t y=0; y<img.height(); y++)
      for (size_t x=0; x<img.width(); x++) {
        size_t idx = getLIdx(x,y,ch,img.width(),img.height());
        T val = img.data_[idx];
        img.data_[idx] = (val < ((T) 0)) ? -val : val;
      }
  return img;
}
//############################################################################
// Explicit instantiations:
template ImageCPU<float> abs( ImageCPU<float> img );





//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 >
T1 sum(const ImageCPU<T1> &img, const ImageCPU<T2> &mask) {
  T1 S = (T1)0;
  for (size_t ch=0; ch<img.channels(); ch++)
    for (size_t y=0; y<img.height(); y++)
      for (size_t x=0; x<img.width(); x++)
        if (mask.data_ == NULL || mask.data_[x+y*mask.width()]) {
          size_t idx = getLIdx(x,y,ch,img.width(),img.height());
          S += img.data_[idx];
        }
  return S;
}
//############################################################################
// Explicit instantiations:
template float sum(const ImageCPU<float> &img, const ImageCPU<bool> &mask);


//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 >
float norm(const ImageCPU<T1> &img, const ImageCPU<T2> &mask,
           std::string norm = "L1") {
  if (norm.compare("L1") == 0) return sum(abs(img), mask);
  else if (norm.compare("L2") == 0) return sqrt(sum(img*img, mask));
  else throw Exception("Error in norm(): Unknown norm %s", norm.c_str());
}
//############################################################################
// Explicit instantiations:
template float norm(const ImageCPU<float> &img, const ImageCPU<bool> &mask,
std::string norm);


//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 >
void applyMask(ImageCPU<T1> &img, const ImageCPU<T2> &mask) {
  int width = img.width();
  int height = img.height();

  if (mask.width() != width || mask.height() != height)
    throw Exception("Error in applyMask(): mask should have same size as input ImageCPU<T> img");
  if (mask.channels() != 1)
    throw Exception ("Error in applyMask(): mask should be one channel only. It will be applies per channel to input ImageCPU<T> img.");

  for (size_t ch=0; ch<img.channels(); ch++)
    for (size_t y=0; y<height; y++)
      for (size_t x=0; x<width; x++) {
        size_t idx_img = getLIdx(x,y,ch,width,height);
        size_t idx_mask = getLIdx(x,y,0,width,height);
        if (!mask.data_[idx_mask]) img.data_[idx_img] = (T1)0;
      }
}
//############################################################################
// Explicit instantiations:
template void applyMask(ImageCPU<float> &img, const ImageCPU<float> &mask);
template void applyMask(ImageCPU<float> &img, const ImageCPU<bool> &mask);



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void getData(const ImageCPU<T> &img, T *arr, std::string aligned = "row-major",
             size_t  step_y = 0, size_t  step_x = 0) {
  size_t channels = img.channels();
  size_t width = img.width();
  size_t height = img.height();
  size_t pixels = width*height*channels;

  if(aligned.compare("row-major") == 0) {
    std::memcpy(arr, img.data_, sizeof(T)*pixels);
  } else if (aligned.compare("opencv") == 0) {
    if (step_x == 0 && step_y == 0)
      throw Exception("If alignment shall be OpenCV, you need to provide step sizes!");
    alignDataCPU2OpCV( img.data_, width, height, channels, arr, step_y, step_x);
  } else if(aligned.compare("matlab") == 0 ||
            aligned.compare("colum-major") == 0) {
    alignDataCPU2Matlab( img.data_, width, height, channels, arr);
  } else
    throw Exception("Error in getData(): Unknown alignment %s", aligned.c_str());
}
//############################################################################
// Explicit instantiations:
template
void getData( const ImageCPU<float> &img, float *arr, std::string aligned, size_t  step_y, size_t  step_x );



//############################################################################
//############################################################################
//############################################################################
#ifdef USE_OPENCV
template< typename T >
void image2OpenCV(const ImageCPU<T> &img, cv::Mat &mat) {
  size_t channels = img.channels();
  size_t width = img.width();
  size_t height = img.height();
  size_t pixels = img.Pixels();

  if(channels >= 4)
    throw Exception("This amount of channels_ %d, is not yet implemented. Maybe use copyDeviceToHost with std::vector<cv::Mat> as output", channels);

  int data_type = cv::DataDepth<T>::value + (channels-1)*8;

  mat.create(height, width, data_type);

  T *p_data = img.data_;
  T *p_mat = (T*) mat.data;

  if (mat.channels() == 1) std::memcpy(p_mat,p_data,sizeof(T)*pixels);
  else getData( img, p_mat, "opencv", mat.step1(0), mat.step1(1) );
}
//############################################################################
// Explicit instantiations:
template void image2OpenCV( const ImageCPU<float> &img, cv::Mat &mat );
template void image2OpenCV( const ImageCPU<bool>  &img, cv::Mat &mat );
#endif



//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T>
//  void get3Dpoints( const ImageCPU<T> &depth,
//      const T *         intrinsics_hst,
//      ImageCPU<T>       &pts_3d)
//  {
//    T* intrinsics_dvc = NULL;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&intrinsics_dvc, 6*sizeof(T)));
//    CUDA_SAFE_CALL(cudaMemcpy((void*)intrinsics_dvc, (void*) intrinsics_hst, 6*sizeof(T), cudaMemcpyHostToDevice));

//    pts_3d  = ImageCPU<float>( depth.width_, depth.height_, 3, 0.f );

//    create3Dpoints( depth.data_, intrinsics_dvc, depth.width_, depth.height_, pts_3d.data_ );

//    CUDA_SAFE_CALL(cudaFree(intrinsics_dvc));

//  }
//  //############################################################################
//  // Explicit instantiations:
//  template void get3Dpoints( const ImageCPU<float> &depth,
//      const float *         intrinsics,
//      ImageCPU<float>       &pts_3d);




//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T >
//  void getChannel( const ImageCPU<T> &img, size_t channel, ImageCPU<T> &chn )
//  {
//    if ( (int)channel >= img.channels_ )
//      return;

//    size_t width_ = img.width_;
//    size_t height_ = img.height_;

//    const size_t    img_size = width_ * height_;

//    chn.release();
//    chn = ImageCPU<T>(width_, height_, 1, &(img.data_[channel*img_size]), true);

//    return;
//  }
//  //############################################################################
//  // Explicit instantiations:
//  template
//  void getChannel( const ImageCPU<float>      &img,
//      size_t          channel,
//      ImageCPU<float> &chn  );




//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T>
//  void getNormals( const ImageCPU<T> &depth,
//      const T *         intrinsics_hst,
//      ImageCPU<T>       &normals)
//  {
//    T* intrinsics_dvc = NULL;
//    CUDA_SAFE_CALL(cudaMalloc((void**)&intrinsics_dvc, 6*sizeof(T)));
//    CUDA_SAFE_CALL(cudaMemcpy((void*)intrinsics_dvc, (void*) intrinsics_hst, 6*sizeof(T), cudaMemcpyHostToDevice));

//    normals = ImageCPU<float>( depth.width_, depth.height_, 3, 0.f );

//    createNormals( depth.data_, intrinsics_dvc, depth.width_, depth.height_, normals.data_);

//    CUDA_SAFE_CALL(cudaFree(intrinsics_dvc));

//  }




//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T>
//  void getNormals( const ImageCPU<T> &depth,
//      ImageCPU<T>       &normals)
//  {
//    normals = ImageCPU<float>( depth.width_, depth.height_, 3, 0.f );

//    createNormals( depth.data_, depth.width_, depth.height_, normals.data_);

//  }


//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T >
//  void inv( ImageCPU<T> &img )
//  {
//    if (img.channels_ == 1)
//      invertElements2D( img.data_, img.width_, img.height_ );
//    else
//      invertElements3D( img.data_, img.width_, img.height_, img.channels_ );
//  }
//  //############################################################################
//  // Explicit instantiations:
//  template void inv( ImageCPU<float> &img );


//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T >
//  void mul( ImageCPU<T> &img, T val )
//  {
//    if (img.channels_ == 1)
//      mulArray2D( img.data_, img.width_, img.height_, val );
//    else
//      mulArray3D( img.data_, img.width_, img.height_, img.channels_, val );
//  }
//  //############################################################################
//  // Explicit instantiations:
//  template void mul( ImageCPU<float> &img, float val );


//  //############################################################################
//  //############################################################################
//  //############################################################################
//  template< typename T >
//  void pov( ImageCPU<T> &img, T exponent )
//  {
//    if (exponent == (T) 1) return;

//    if (img.channels_ == 1)
//      povArray2D( img.data_, img.width_, img.height_, exponent );
//    else
//      povArray3D( img.data_, img.width_, img.height_, img.channels_, exponent );
//  }
//  //############################################################################
//  // Explicit instantiations:
//  template void pov( ImageCPU<float> &img, float exponent );




//############################################################################
//############################################################################
//############################################################################
template< typename T >
void saveImage(const std::string &filename, const ImageCPU<T> &img) {
  if ( img.data_ == NULL )
    throw Exception( "Error saveImage(): no data_ in img\n" );
  StoreArrayAsBinary(filename, img.data_,
                     img.width(), img.height(), img.channels());
}
//############################################################################
// Explicit instantiations:
template
void saveImage( const std::string &filename, const ImageCPU<float> &img  );



//############################################################################
//############################################################################
//############################################################################
template< typename T >
void saveImageChannel(const std::string &filename, const ImageCPU<T> &img,
                      size_t channel) {
  if ( img.data_ == NULL )
    throw Exception( "Error saveImage(): no data_ in img\n" );
  StoreArrayAsBinary(filename, &img.data_[channel*img.width()*img.height()],
      img.width(), img.height(), 1);
}
//############################################################################
// Explicit instantiations:
template
void saveImageChannel( const std::string &filename, const ImageCPU<float> &img, size_t channel  );



//############################################################################
//############################################################################
//############################################################################
#ifdef USE_OPENCV
template< typename T1, typename T2 = T1 >
void imshow(const std::string &window_name, const ImageCPU<T1> &img,
            const T2 &factor = (T2) 1) {
  cv::Mat mat;
  image2OpenCV(img, mat);
  cv::imshow(window_name, factor * mat);
}
//############################################################################
// Explicit instantiations:
template void imshow( const std::string &window_name, const ImageCPU<float> &img, const float &factor );
template void imshow( const std::string &window_name, const ImageCPU<bool> &img, const float &factor );




//############################################################################
//############################################################################
//############################################################################
template< typename T1, typename T2 = T1 >
void imwrite(const std::string &file_name, const ImageCPU<T1> &img,
             const T2 &factor = (T2) 255) {
  cv::Mat mat;
  image2OpenCV(img, mat);
  cv::imwrite(file_name, factor * mat);
}
//############################################################################
// Explicit instantiations:
template void imwrite(const std::string &file_name, const ImageCPU<float> &img, const float &factor );
template void imwrite( const std::string &file_name, const ImageCPU<bool> &img, const float &factor );
#endif //USE_OPENCV

#endif  // MUMFORDSHAH_LIB_IMAGE_CPU_H_
