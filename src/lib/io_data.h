/*
 * IOData.hpp
 *
 *  Created on: Jun 21, 2017
 *      Author: haefner
 */

#ifndef MUMFORDSHAH_LIB_IODATA_H_
#define MUMFORDSHAH_LIB_IODATA_H_

#include <stddef.h>
#include <string>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

bool LoadRgb(const std::string &filename, cv::Mat &rgb,
             float factor = 1.0f / 255.0f);
bool LoadMask(const std::string &filename, cv::Mat &mask);
bool StoreCVMatAsBinary(const std::string &filename, const cv::Mat &mat);
#endif

template <typename T>
bool StoreArrayAsBinary(const std::string &filename,
                        const T *data,
                        const size_t &width,
                        const size_t &height,
                        const size_t &channels);

#endif /* MUMFORDSHAH_LIB_IODATA_H_ */
