/*
 * IOData.cpp
 *
 *  Created on: Jun 21, 2017
 *      Author: haefner
 */
#include "io_data.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


#ifdef USE_OPENCV
bool LoadMask(const std::string &filename, cv::Mat &mask) {
  if (filename.empty()) {
    std::cout << "Error in loadMask(): Input string empty." << std::endl;
    return false;
  }
  mask = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  if (mask.type() != CV_8U || mask.channels() != 1)
    mask.convertTo(mask, CV_8U);
  mask /= 255.f; //to get "boolean" values
  std::cout << "Successfully loaded: " << filename << std::endl;
  return true;
}


bool LoadRgb(const std::string &filename, cv::Mat &rgb,
             float factor /*= 1.0f / 255.0f*/) {
  if (filename.empty()) {
    std::cout << "Error in loadRgb(): Input string empty." << std::endl;
    return false;
  }
  rgb = cv::imread(filename);
  // convert 8bit to float
  if (rgb.type() != CV_32F || rgb.channels() != 3)
    rgb.convertTo(rgb, CV_32FC3, factor);
  std::cout << "Successfully loaded: " << filename << "\n";
  return true;
}


bool StoreCVMatAsBinary(const std::string &filename, const cv::Mat &mat) {
  if (filename.empty())
    return false;

  // store float depth to binary file
  std::ofstream out_file;
  out_file.open(filename.c_str(), std::ios::binary);
  if (!out_file.is_open())
    return false;
  // write depth data
  out_file.write((const char*)mat.data,
                 mat.elemSize1()*mat.cols*mat.rows*mat.channels());  // depth.elemSize1() is e.g. sizeof(float), sizeof(int), etc.
  out_file.close();
  return true;
}
#endif //USE_OPENCV


template <typename T>
bool StoreArrayAsBinary(const std::string &filename,
                        const T *data,
                        const size_t &width,
                        const size_t &height,
                        const size_t &channels) {
  if (filename.empty())
    return false;

  // store float depth to binary file
  std::ofstream out_file;
  out_file.open(filename.c_str(), std::ios::binary);
  if (!out_file.is_open())
    return false;
  // write depth data
  size_t size = width * height * channels;
  out_file.write((const char*)data, sizeof(T) * size);
  out_file.close();
  return true;
}//############################################################################
// Explicit instantiations:
template bool StoreArrayAsBinary(const std::string &filename, const float *data, const size_t &width, const size_t &height, const size_t &channels);
