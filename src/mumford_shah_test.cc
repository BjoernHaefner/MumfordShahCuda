#include <iostream>

#include <opencv2/core/core.hpp>

#include "lib/mumford_shah.h"
#include "lib/image.h"

//Activate Visual Leak Detector
#if ( defined(WIN32) || defined(WIN64) ) && defined(_DEBUG)
//  #include <vld.h>
#endif

//to use cmake defined variable APP_SOURCE_DIR
#define MAKE_STRING_(x)  #x
#define MAKE_STRING(x)  MAKE_STRING_(x)

bool TestMumfordShahVersion1CommandLine(int argc, const char *argv[]);
bool TestMumfordShahVersion2UseString(std::string data,
                                      std::string img_file,
                                      std::string scalar_op_file,
                                      std::string mask_file,
                                      std::string output_prefix,
                                      float lambda,
                                      float alpha,
                                      float tol,
                                      int max_iter,
                                      bool verbose,
                                      bool show_result,
                                      bool show);
bool TestMumfordShahVersion3PureOpenCV(std::string data,
                                       std::string img_file,
                                       std::string scalar_op_file,
                                       std::string mask_file,
                                       float lambda,
                                       float alpha,
                                       float tol,
                                       int max_iter,
                                       bool verbose,
                                       bool show_result,
                                       bool show);
bool TestMumfordShahVersion4Pointer(std::string data,
                                    std::string img_file,
                                    std::string scalar_op_file,
                                    std::string mask_file,
                                    float lambda,
                                    float alpha,
                                    float tol,
                                    int max_iter,
                                    bool verbose,
                                    bool show_result,
                                    bool show);
bool TestMumfordShahVersion5MultipleRuns(std::string data,
                                         std::string img_file,
                                         std::string scalar_op_file,
                                         std::string mask_file,
                                         float lambda,
                                         float alpha,
                                         float tol,
                                         int max_iter,
                                         bool verbose,
                                         bool show_result,
                                         bool show);

int main(int argc, const char *argv[]) {

  // test mumfordshah with command line parameters
//  std::cout << "Start algorithm with command line parameters" << std::endl;
//  TestMumfordShahVersion1CommandLine(argc,argv);

//  //run remaining code with hardcoded parameters
  std::string data = std::string(MAKE_STRING(APP_SOURCE_DIR))+ "/data/";
  std::string img_file = "intensity.png";
  std::string scalar_op_file = "shading.png";
  std::string mask_file = "mask.png";
  std::string output_prefix = "result.png";
  float lambda = 0.1f;
  float alpha = -1.f;
  float tol = 1e-5f;
  int max_iter = 5000;
  bool verbose = true;
  bool show_result = true;
  bool show = false;
  // test mumfordshah with hard coded parameters
  std::cout << std::endl << std::endl;
  std::cout << "Start algorithm with strings" << std::endl;
  TestMumfordShahVersion2UseString(data, img_file, scalar_op_file, mask_file,
                                   output_prefix,
                                   lambda, alpha, tol, max_iter,
                                   verbose, show_result, show);
  std::cout << std::endl << std::endl;
  std::cout << "Start algorithm with OpenCV matrices" << std::endl;
  TestMumfordShahVersion3PureOpenCV(data, img_file, scalar_op_file, mask_file,
                                    lambda, alpha, tol, max_iter,
                                    verbose, show_result, show);
  std::cout << std::endl << std::endl;
  std::cout << "Start algorithm with pointer arrays" << std::endl;
  TestMumfordShahVersion4Pointer(data, img_file, scalar_op_file, mask_file,
                                 lambda, alpha, tol, max_iter,
                                 verbose, show_result, show);
  // test mumfordshah with multiple runs and varying scalar operator
  std::cout << std::endl << std::endl;
  std::cout << "Start algorithm for multiple runs with same data" << std::endl;
  TestMumfordShahVersion5MultipleRuns(data, img_file, scalar_op_file, mask_file,
                                      lambda, alpha, tol, max_iter,
                                      verbose, show_result, show);
}


bool TestMumfordShahVersion1CommandLine(int argc, const char *argv[]) {
  MumfordShah ms(argc, argv);
  if (!ms.Run()) {
    std::cout << "Exit program. Unsuccessfull" << std::endl;
    return 0;
  } else {
    std::cout << "Exit program. Successfull" << std::endl;
    return 1;
  }
}


bool TestMumfordShahVersion2UseString(std::string data,
                                      std::string img_file,
                                      std::string scalar_op_file,
                                      std::string mask_file,
                                      std::string output_prefix,
                                      float lambda,
                                      float alpha,
                                      float tol,
                                      int max_iter,
                                      bool verbose,
                                      bool show_result,
                                      bool show) {
  MumfordShah ms;
  // data
  ms.set_lambda(lambda);
  ms.set_alpha(alpha); //-1 means piecewise constant);
  ms.set_max_iter(max_iter);
  ms.set_tol(tol);
  ms.set_verbose(verbose);
  ms.set_show_result(show_result);
  ms.set_show(show);
  if (!ms.Run(data + img_file, data + scalar_op_file,
              data + mask_file, data + output_prefix)) {
    std::cout << "Exit program. Unsuccessfull" << std::endl;
    return 0;
  } else {
    std::cout << "Exit program. Successfull" << std::endl;
    return 1;
  }
}


bool TestMumfordShahVersion3PureOpenCV(std::string data,
                                       std::string img_file,
                                       std::string scalar_op_file,
                                       std::string mask_file,
                                       float lambda,
                                       float alpha,
                                       float tol,
                                       int max_iter,
                                       bool verbose,
                                       bool show_result,
                                       bool show) {
  MumfordShah ms;
  // data
  ms.set_lambda(lambda);
  ms.set_alpha(alpha); //-1 means piecewise constant);
  ms.set_max_iter(max_iter);
  ms.set_tol(tol);
  ms.set_verbose(verbose);
  ms.set_show_result(show_result);
  ms.set_show(show);
  //load depths files
  cv::Mat img;
  if (!LoadRgb(data + img_file, img)) {
    std::cout << "Could not load rgb file at '" <<
                 img_file << "'." << std::endl;
    return false;
  }

  //load scalar operator
  cv::Mat scalar_op;
  if (!LoadRgb(data + scalar_op_file, scalar_op)) {
    std::cout << "Could not load scalar_op file at '" <<
                 scalar_op_file << "'." << std::endl;
    return false;
  }

  //load mask
  cv::Mat mask;
  if (!LoadMask(data + mask_file, mask)) {
    std::cout << "Could not load mask file at '" <<
                 mask_file << "'." << std::endl;
    return false;
  }

  cv::Mat cv_result;
  if (!ms.Run(img, scalar_op, mask, cv_result)) {
    std::cout << "Exit program. Unsuccessfull" << std::endl;
    return 0;
  } else {
    std::cout << "Show output." << std::endl;
    cv::imshow("result", cv_result);
    std::cout << "Press any key to continue/finish." << std::endl;
    cv::waitKey();
    cv::destroyWindow("result");
    std::cout << "Exit program. Successfull" << std::endl;
    return 1;
  }
}


bool TestMumfordShahVersion4Pointer(std::string data,
                                    std::string img_file,
                                    std::string scalar_op_file,
                                    std::string mask_file,
                                    float lambda,
                                    float alpha,
                                    float tol,
                                    int max_iter,
                                    bool verbose,
                                    bool show_result,
                                    bool show) {
  MumfordShah ms;
  // params
  ms.set_lambda(lambda);
  ms.set_alpha(alpha); //-1 means piecewise constant);
  ms.set_max_iter(max_iter);
  ms.set_tol(tol);
  ms.set_verbose(verbose);
  ms.set_show_result(show_result);
  ms.set_show(show);

  //load data
  cv::Mat img_cv;
  if (!LoadRgb(data + img_file, img_cv)) {
    std::cout << "Could not load rgb file at '" <<
                 img_file << "'." << std::endl;
    return false;
  }
  Image<float> img(img_cv);

  //load scalar operator
  cv::Mat scalar_op_cv;
  if (!LoadRgb(data + scalar_op_file, scalar_op_cv)) {
    std::cout << "Could not load scalar_op file at '" <<
                 scalar_op_file << "'." << std::endl;
    return false;
  }
  Image<float> scalar_op(scalar_op_cv);

  //load mask
  cv::Mat mask_cv;
  if (!LoadMask(data + mask_file, mask_cv)) {
    std::cout << "Could not load mask file at '" <<
                 mask_file << "'." << std::endl;
    return false;
  }
  Image<bool> mask(mask_cv);

  Image<float> result(img.width(), img.height(), img.channels());
  if (!ms.Run(img.data_, scalar_op.data_, mask.data_,
              img.width(),img.height(),img.channels(),
              result.data_,"row-major")) {
    std::cout << "Exit program. Unsuccessfull" << std::endl;
    return 0;
  } else {
    std::cout << "Show output." << std::endl;
    imshow("result", result);
    std::cout << "Press any key to continue/finish." << std::endl;
    cv::waitKey();
    cv::destroyWindow("result");
    std::cout << "Exit program. Successfull" << std::endl;
    return 1;
  }
}


bool TestMumfordShahVersion5MultipleRuns(std::string data,
                                         std::string img_file,
                                         std::string scalar_op_file,
                                         std::string mask_file,
                                         float lambda,
                                         float alpha,
                                         float tol,
                                         int max_iter,
                                         bool verbose,
                                         bool show_result,
                                         bool show) {
  MumfordShah ms;
  // data
  ms.set_lambda(lambda);
  ms.set_alpha(alpha); //-1 means piecewise constant);
  ms.set_max_iter(max_iter);
  ms.set_tol(tol);
  ms.set_verbose(verbose);
  ms.set_show_result(show_result);
  ms.set_show(show);
  //load depths files
  cv::Mat img_cv;
  if (!LoadRgb(data + img_file, img_cv)) {
    std::cout << "Could not load rgb file at '" <<
                 img_file << "'." << std::endl;
    return false;
  }
  Image<float> img(img_cv);

  //load scalar operator
  cv::Mat scalar_op_cv;
  if (!LoadRgb(data + scalar_op_file, scalar_op_cv)) {
    std::cout << "Could not load scalar_op file at '" <<
                 scalar_op_file << "'." << std::endl;
    return false;
  }
  //HACK
  Image<float> scalar_op(scalar_op_cv);
//  Image<float> scalar_op(img.width(),img.height(),img.channels(),1.f);

  //load mask
  cv::Mat mask_cv;
  if (!LoadMask(data + mask_file, mask_cv)) {
    std::cout << "Could not load mask file at '" <<
                 mask_file << "'." << std::endl;
    return false;
  }
  Image<bool> mask(mask_cv);
  ms.SetData(img.data_, mask.data_,
             img.height(), img.width(), img.channels(), "row-major");

  Image<float> result(img.width(), img.height(), img.channels());
  int n_runs = 10;
  for (int i = 0; i<n_runs; i++) {
    if (!ms.Run(scalar_op.data_, result.data_, "row-major")) {
      std::cout << "Exit program. Unsuccessfull" << std::endl;
      return 0;
    } else {
      std::cout << "Show output of run " <<i<<"/"<<n_runs-1<< "." << std::endl;
      imshow("result", result);
      std::cout << "Press any key to continue." << std::endl << std::endl;
      cv::waitKey(1000);
    }
  }
  std::cout << "Press any key to finish." << std::endl;
  cv::waitKey();
  cv::destroyWindow("result");
  std::cout << "Exit program. Successfull" << std::endl;
  return 1;
}
