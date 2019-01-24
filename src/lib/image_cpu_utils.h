#ifndef MUMFORDSHAH_LIB_IMAGE_CPU_UTILS_H_
#define MUMFORDSHAH_LIB_IMAGE_CPU_UTILS_H_

#include <stddef.h>

size_t inline getLIdx(size_t x, size_t y, size_t ch,
                      size_t width, size_t height) {
  return x + y * width  + ch * width * height;
}
size_t inline getLIdxColumMajor(size_t x, size_t y, size_t ch,
                                size_t width, size_t height) {
  return y + x * height + ch * width * height;
}
size_t inline getLIdxMatlab(size_t x, size_t y, size_t ch,
                            size_t width, size_t height) {
  return getLIdxColumMajor(x, y, ch, width, height);
}
size_t inline getLIdxOpenCV(size_t x, size_t y, size_t ch,
                            size_t step_x, size_t step_y) {
  return ch + x * step_x + y * step_y;
}

//  /**
//   * \brief  Fills an array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  dataSize  Number of elements in the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  addToArray1D( T   *arr,
//      int length,
//      T   val );


//  /**
//   * \brief  Fills an array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  dataSize  Number of elements in the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  addToArray2D( T   *arr_2d,
//      int width,
//      int height,
//      T   val );


//  /**
//   * \brief  Fills an array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  dataSize  Number of elements in the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  addToArray3D( T   *arr_3d,
//      int width,
//      int height,
//      int depth,
//      T   val );


  /**
   * \brief  Fills an array with a constant value
   *
   * \param  arr       The array to fill
   * \param  dataSize  Number of elements in the array
   * \param  value     The value to fill the array with
   */
  template<typename T>
  void  addToArray( T   *arr1,
      int width,
      int height,
      int depth,
      const T   *arr2 );


  /**
   * TODO
   */
  template<typename T>
  void  divArray( T      *num,
      const T * denom,
      int    width,
      int    height,
      int    depth );


  /**
   * TODO
   */
  template<typename T>
  void  divArray( T      *num,
      const T denom,
      int    width,
      int    height,
      int    depth );


  /**
   * TODO
   */
  template<typename T>
  void  divArray( T num,
      const T *denom,
      int    width,
      int    height,
      int    depth);


  /**
   * \brief  Fills an array with a constant value
   *
   * \param  arr       The array to fill
   * \param  dataSize  Number of elements in the array
   * \param  value     The value to fill the array with
   */
  template<typename T>
  void  subFromArray( T   *arr1,
      int width,
      int height,
      int depth,
      const T   *arr2 );





  /**/
  template< typename T1, typename T2 >
   void alignDataCPU2Matlab( const T1 * data,
       size_t width,
       size_t height,
       size_t channels,
       T2 * matlab);





  /**/
  template< typename T1, typename T2 >
  void alignDataCPU2OpCV( const T1 *     data,
      size_t  width,
      size_t  height,
      size_t  channels,
      T2 *     opcv,
      size_t  step_y,
      size_t  step_x );





  /**/
  template< typename T1, typename T2 >
   void alignDataMatlab2CPU( T1 * data,
       size_t width,
       size_t height,
       size_t channels,
       const T2 * matlab);





  /**/
  template< typename T1, typename T2 >
  void alignDataOpCV2CPU( T1 * data,
      size_t  width,
      size_t  height,
      size_t  channels,
      const T2 * opcv,
      size_t  step_y,
      size_t  step_x );



//  /**
//   * \brief  Apply a mask to a corresponding image, setting the corresponding
//   *         pixels to a certain value
//   *
//   * \param[in,out] ptr_img             The image to change
//   * \param         ptr_mask            The mask
//   * \param         width               Width of the image
//   * \param         height              Height of the image
//   * \param         channels            Number of channels
//   */
//  template< typename T1, typename T2 >
//  void apply2DMask( T1*  ptr_img,
//      int            width,
//      int            height,
//      int            channels,
//      const T2   *  ptr_mask );

//  template< typename T>
//  void create3Dpoints(  const T * depth,
//      const T * intrinsics_dvc,
//      int       width,
//      int       height,
//      T       * pts_3d);


//  template< typename T>
//  void createNormals( const T * depth,
//      const T * intrinsics_dvc,
//      int       width,
//      int       height,
//      T       * normals);


//  template< typename T>
//  void createNormals( const T * depth,
//      int       width,
//      int       height,
//      T       * normals);


//  /**
//   * \brief  Inverts the data elements:
//   *           if ( data[i] < 1e-9f )
//   *             data[i] = 0;
//   *           else
//   *             data[i] = 1.0f / data[i];
//   *
//   * \param  data      The data elements to invert
//   * \param  dataSize  Number of data elements
//   */
//  template<typename T>
//  void  invertElements1D( T  *data,
//      int    length );


//  /**
//   * \brief  Inverts the data elements:
//   *           if ( data[i] < 1e-9f )
//   *             data[i] = 0;
//   *           else
//   *             data[i] = 1.0f / data[i];
//   *
//   * \param  data      The data elements to invert
//   * \param  dataSize  Number of data elements
//   */
//  template<typename T>
//  void  invertElements2D( T  *arr_2d,
//      int     width,
//      int     height);


//  /**
//   * \brief  Inverts the data elements:
//   *           if ( data[i] < 1e-9f )
//   *             data[i] = 0;
//   *           else
//   *             data[i] = 1.0f / data[i];
//   *
//   * \param  data      The data elements to invert
//   * \param  dataSize  Number of data elements
//   */
//  template<typename T>
//  void  invertElements3D( T  *arr_3d,
//      int     width,
//      int     height,
//      int     depth);


  /**
   * \brief  Multiply each element of a 3d array/cube with the given value
   *
   * \param  data      The 3d array/cube to modify
   * \param  width     Width of the 3d array/cube
   * \param  height    Height of the 3d array/cube
   * \param  depth     Depth of the 3d array/cube
   * \param  value     The value to multiply
   */
  template<typename T>
  void  mulArray( T      *arr,
      int    width,
      int    height,
      int    depth,
      T      value );


  /**
   * TODO
   */
  template<typename T>
  void  mulArray( T      *arr,
      const T * arr2,
      int    width,
      int    height,
      int    depth );


//  /**
//   * \brief  TODO
//   */
//  template<typename T>
//  T   norm2D( const T   *arr_2d,
//        int     width,
//        int     height,
//        std::string norm = "L2");




//  /**
//   * \brief  TODO
//   */
//  template<typename T>
//    void  povArray1D( T      *arr,
//        int    length,
//        T      exponent );




//  /**
//   * \brief  TODO
//   */
//  template<typename T>
//    void  povArray2D( T      *arr,
//        int    width,
//        int    height,
//        T      exponent );




//  /**
//   * \brief  TODO
//   */
//  template<typename T>
//    void  povArray3D( T      *arr,
//        int     width,
//        int    height,
//        int    depth,
//        T      exponent );




//  /**
//   * \brief  print 1D array value
//   *
//   * \param      arr
//   * \param      pos     position to be printed
//   * \param      size    Number of array elements
//   */
//  template< typename T >
//  void   printArray1D( const T  *arr,
//      int          pos,
//      int          size );





//  /**
//   * \brief  print 2D array value
//   *
//   * \param      arr
//   * \param      pos_x   x position to be printed
//   * \param      pos_y   y position to be printed
//   * \param      width   width of 2D array
//   * \param      height  height of 2D array
//   */
//  template< typename T >
//  void printArray2D( const T   *arr,
//      int     pos_x,
//      int     pos_y,
//      int     width,
//      int     height );





//  /**
//   * \brief  print 2D array value
//   *
//   * \param      arr
//   * \param      pos    linear position to be printed
//   * \param      width  width of 2D array
//   * \param      height height of 2D array
//   */
//  template< typename T >
//  void printArray2D( const T   *arr,
//      int     pos,
//      int     width,
//      int     height );





//  /**
//   * \brief  print 3D array value
//   *
//   * \param      arr
//   * \param      pos_x    x position to be printed
//   * \param      pos_y    y position to be printed
//   * \param      pos_z    z position to be printed
//   * \param      width    width of 2D array
//   * \param      height   height of 2D array
//   * \param      depth    depth of 2D array
//   */
//  template< typename T >
//  void printArray3D( const T   *arr,
//      int     pos_x,
//      int     pos_y,
//      int     pos_z,
//      int     width,
//      int     height,
//      int     depth);





//  /**
//   * \brief  print 3D array value
//   *
//   * \param      arr
//   * \param      pos    linear position to be printed
//   * \param      width  width of 3D array
//   * \param      height height of 3D array
//   * \param      depth  depth of 3D array
//   */
//  template< typename T >
//  void printArray3D( const T   *arr,
//      int     pos,
//      int     width,
//      int     height,
//      int     depth);


//  /**
//   * \brief  Fills an array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  dataSize  Number of elements in the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  setArray1D( T    *arr,
//      size_t  length,
//      T    value );


//  /**
//   * \brief  Fills a 2D array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  width     width of the array
//   * \param  height    height of the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  setArray2D( T    *arr_2d,
//      size_t  width,
//      size_t  height,
//      T    value );


//  /**
//   * \brief  Fills a 2D array with a constant value
//   *
//   * \param  arr       The array to fill
//   * \param  width     width of the array
//   * \param  height    height of the array
//   * \param  value     The value to fill the array with
//   */
//  template<typename T>
//  void  setArray3D( T    *arr_3d,
//      size_t  width,
//      size_t  height,
//      size_t  depth,
//      T    value );




//  /**
//   * \brief  Sums up all elements of the given 2D device array
//   *
//   * \param  d_data  The 2D array in device memory - Warning: Gets modified!
//   * \param  width   Width of the 2D array
//   * \param  height  Height of the 2D array
//   * \param  xoff    x-offset
//   * \param  yoff    y-offset
//   * \param  stride  Stride (= Memory aligned width in elements)
//   *
//   * \return Sum of all elements
//   */
//  template<typename T>
//  T  sum2D( const T   *arr_2d,
//      int     width,
//      int     height );

//}
#endif /* _IMAGECPUUTILS_H_ */
