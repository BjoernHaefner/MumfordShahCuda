#include "mumford_shah.h"

//stl
#include <vector>
#include <iostream>
#include <math.h> //sqrt

//opencv
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif //USE_OPENCV

//CUDA
#ifdef USE_CUDA
#include "cuda/cuda_common.cuh"
#endif //USE_CUDA

//local code (used to load data using opencv)
#include "io_data.h"


MumfordShah::MumfordShah() {
#ifdef USE_CUDA
  CUDA::GPUinit();
#endif //USE_CUDA
}


#ifdef USE_OPENCV
MumfordShah::MumfordShah(int argc, const char *argv[]) {
#ifdef USE_CUDA
  CUDA::GPUinit();
#endif // USE_CUDA
  InputArgumentHandling(argc, argv);
}
#endif //USE_OPENCV


MumfordShah::~MumfordShah() {
#ifdef USE_CUDA
  CUDA::GPUclose();
#endif //USE_CUDA
}


#ifdef DEBUG
void MumfordShah::Show() {
  imshow(std::string("intensity"), data_.intensity);
  imshow(std::string("scalar_op"), data_.scalar_op);
  imshow(std::string("mask"), data_.mask, 255.f);

  if (pd_algo_.u.data_ != NULL) {
    imshow(std::string("u"),pd_algo_.u);
    imshow(std::string("u_bar"), pd_algo_.u_bar);

    Image<float> p1(pd_algo_.p.width(),
                    pd_algo_.p.height(),
                    pd_algo_.p.channels()/2,
                    &(pd_algo_.p.data_[0]));
    Image<float> p2(pd_algo_.p.width(),
                    pd_algo_.p.height(),
                    pd_algo_.p.channels()/2,
                    &(pd_algo_.p.data_[pd_algo_.p.Pixels()/2]));

    imshow(std::string("p1"), p1);
    imshow(std::string("p2"), p2);
  }
}


void MumfordShah::Write() {
  std::string data_path = "/usr/wiss/haefner/Documents/coding/cpp/mumfordshah/data/";

  imwrite(std::string(data_path + std::string("intensity_debug.png")),
          data_.intensity);
  imwrite(std::string(data_path + std::string("scalar_op_debug.png")),
          data_.scalar_op);
  imwrite(std::string(data_path + std::string("mask_debug.png")),
          data_.mask);

  if (pd_algo_.u.data_ != NULL) {
    imwrite(std::string(data_path + std::string("algo_u.png")),
            pd_algo_.u);
    imwrite(std::string(data_path + std::string("algo_u_bar.png")),
            pd_algo_.u_bar);

    Image<float> p1(pd_algo_.p.width(),
                    pd_algo_.p.height(),
                    pd_algo_.p.channels()/2,
                    &(pd_algo_.p.data_[0]));
    Image<float> p2(pd_algo_.p.width(),
                    pd_algo_.p.height(),
                    pd_algo_.p.channels()/2,
                    &(pd_algo_.p.data_[pd_algo_.p.Pixels()/2]));

    imwrite(std::string(data_path + std::string("algo_p1.png")), p1, 255);
    imwrite(std::string(data_path + std::string("algo_p2.png")), p2, 255);
  }
}


void MumfordShah::Debug(const std::string &message) {
  std::cout << message << std::endl;
  Show( );
  //  Write( );

  std::cout << "Press any key to continue." << std::endl;
  cv::waitKey();
  cv::destroyAllWindows();
}
#endif  // DEBUG


#ifdef USE_OPENCV
bool MumfordShah::Run(float *result) {
  applyMask(data_.intensity, data_.mask);
  applyMask(data_.scalar_op, data_.mask);

  InitMumfordShah();
  PDAlgorithm();

  if (result != NULL) {
    getData(pd_algo_.u,result);
  }
  if (!data_.output_prefix.empty()) {
    imwrite(data_.output_prefix,pd_algo_.u);
  }
  if(params_.show_result) {
    std::cout << "Show output." << std::endl;
    imshow("result", pd_algo_.u);
    std::cout << "Press any key to continue/finish." << std::endl;
    cv::waitKey();
    cv::destroyWindow("result");
  }
  return true;
}


bool MumfordShah::Run(const std::string &rgb_file,
                      const std::string &scalar_op_file,
                      const std::string &mask_file,
                      const std::string &output_file) {
  //load depths files
  cv::Mat rgb;
  if (!LoadRgb(rgb_file, rgb)) {
    std::cout << "Could not load rgb file at '" <<
                 rgb_file << "'." << std::endl;
    return false;
  }
  //  imwrite("/usr/wiss/haefner/Documents/coding/cpp/mumfordshah/data/intensity_out.png",rgb);

  //load scalar operator
  cv::Mat scalar_op;
  if (!LoadRgb(scalar_op_file, scalar_op)) {
    std::cout << "Could not load scalar_op file at '" <<
                 scalar_op_file << "'." << std::endl;
    return false;
  }
  //  imwrite("/usr/wiss/haefner/Documents/coding/cpp/mumfordshah/data/shading_out.png",scalar_op);


  //load mask
  cv::Mat mask;
  if (!LoadMask(mask_file, mask)) {
    std::cout << "Could not load mask file at '" <<
                 mask_file << "'." << std::endl;
    return false;
  }
  //  imwrite("/usr/wiss/haefner/Documents/coding/cpp/mumfordshah/data/mask_out.png",mask);

  cv::Mat cv_result;
  if (!Run(rgb, scalar_op, mask, cv_result)) {
    std::cerr << "Could not process cv::Mat files!" << std::endl;
    return false;
  }

  if (!output_file.empty()) {
    // store rendered depth map
    std::stringstream ss;
    ss << output_file << ".bin";
    if (StoreCVMatAsBinary(ss.str(), cv_result))
      std::cout << "Output is stored as binary in '" <<
                   ss.str() << "'" << std::endl;
    if (cv::imwrite(output_file, 255*cv_result))
      std::cout << "Output is stored as png in '" <<
                   output_file.c_str() << "'" << std::endl;
  }
  if(params_.show_result) {
    std::cout << "Show output." << std::endl;
    cv::imshow("result", cv_result);
    std::cout << "Press any key to continue/finish." << std::endl;
    cv::waitKey();
    cv::destroyWindow("result");
  }

  return true;

}//run


bool MumfordShah::Run(const cv::Mat &rgb, const cv::Mat &scalar_op,
                      const cv::Mat &mask, cv::Mat &cv_result) {
  // Allocate memory for image input data
  data_.intensity = Image<float>(rgb);
  data_.scalar_op = Image<float>(scalar_op);
  data_.mask = Image<bool>(mask);

  applyMask(data_.intensity, data_.mask);
  applyMask(data_.scalar_op, data_.mask);

  InitMumfordShah();

  PDAlgorithm();

  image2OpenCV(pd_algo_.u, cv_result);

  return true;

}//run
#endif //USE_OPENCV


bool MumfordShah::Run(const float *rgb, const float *scalar_op,
                      const bool *mask,
                      const int width, const int height, const int channels,
                      float *ptr_result,
                      const std::string aligned) {
  // Allocate memory for image input data
  data_.intensity = Image<float>(width, height, channels, rgb, aligned);
  data_.scalar_op = Image<float>(width, height, channels, scalar_op, aligned);
  data_.mask = Image<bool>(width, height, 1, mask, aligned );

  applyMask(data_.intensity, data_.mask);
  applyMask(data_.scalar_op, data_.mask);

  InitMumfordShah();
  PDAlgorithm();

  getData(pd_algo_.u, ptr_result, aligned);

  return true;

}//run


bool MumfordShah::SetData(const float* rgb,
                          const bool * mask,
                          const int height, const int width, const int channels,
                          const std::string aligned) {

  // Allocate GPU memory for image input data
  data_.intensity = Image<float>(width, height, channels, rgb, aligned);
  data_.mask = Image<bool >(width, height, 1, mask, aligned);

  applyMask(data_.intensity, data_.mask);

  InitMumfordShah();

  return true;
}


bool MumfordShah::Run(const float *scalar_op,
                      float *ptr_result, const std::string aligned) {
  data_.scalar_op = Image<float>(data_.intensity.width(),
                                 data_.intensity.height(),
                                 data_.intensity.channels(),
                                 scalar_op, aligned);
  applyMask(data_.scalar_op, data_.mask);
  PDAlgorithm();
  if (ptr_result != NULL)
    getData(pd_algo_.u, ptr_result, aligned);
  return true;
} //MumfordShah::Run

//##########################################################################
//################################PRIVATE
//##########################################################################
void  MumfordShah::PrintDetails() const {
  std::cout << "Variables:" << std::endl;
  std::cout << "[width,height,channels] = ["
            << data_.intensity.width() << ", "
            << data_.intensity.height() << ", "
            << data_.intensity.channels() << "]" << std::endl;
  std::cout << "alpha                   = [" << params_.alpha << "]" << std::endl;
  std::cout << "lambda                  = [" << params_.lambda << "]" << std::endl;
  std::cout << "max_iter                = [" << params_.max_iter << "]" << std::endl;
  std::cout << "tol                     = [" << params_.tol << "]" << std::endl;
  std::cout << "verbose                 = [" << params_.verbose << "]" << std::endl;
#ifdef USE_OPENCV
  std::cout << "show_result             = [" << params_.show_result << "]" << std::endl;
  std::cout << "show_all                = [" << params_.show << "]" << std::endl;
#endif
  std::cout << "gamma                   = [" << params_.gamma << "]" << std::endl;
  std::cout << "tau_init                = [" << pd_algo_.tau_init << "]" << std::endl;
  std::cout << "sigma_init              = [" << pd_algo_.sigma_init << "]" << std::endl << std::endl;

}//MumfordShah::PrintDetails


void  MumfordShah::InitMumfordShah() {
  data_.img_domain = 0.f;
  for (int pos=0; pos<data_.mask.Pixels(); pos++)
    data_.img_domain += 1.f;
  data_.img_domain *= data_.intensity.channels();

  //  InitPDAlgorithm();

#ifdef USE_CUDA
  //calculate block size and grid size for sr- and lr-image space
  CUDA::get3DGridBlock(data_.intensity.width(),
                       data_.intensity.height(),
                       data_.intensity.channels(),
                       cuda_params_.dimGrid3D, cuda_params_.dimBlock3D);
  CUDA::get2DGridBlock(data_.intensity.width(),
                       data_.intensity.height(),
                       cuda_params_.dimGrid2D, cuda_params_.dimBlock2D);
#endif
  if (params_.verbose)
    PrintDetails();
}//MumfordShah::InitMumfordShah


void MumfordShah::InitPDAlgorithm() {
  // Initialize dual variable
  pd_algo_.p = Image<float>(data_.intensity.width(),
                            data_.intensity.height(),
                            2*data_.intensity.channels(), 0.f);
  // initialize (primal) variables
  pd_algo_.u = Image<float>(data_.intensity);
  pd_algo_.u_bar = Image<float>(data_.intensity);
  pd_algo_.u_diff = Image<float>(data_.intensity.width(),
                                 data_.intensity.height(),
                                 data_.intensity.channels(), 0.f);
  // initialize step sizes
  pd_algo_.tau = pd_algo_.tau_init;
  pd_algo_.sigma = pd_algo_.sigma_init;
}//MumfordShah::InitPDAlgorithm


void MumfordShah::PDAlgorithm() {
  InitPDAlgorithm();
  //start iterations
  for (int ii_iter=0; ii_iter<params_.max_iter; ii_iter++) {
    UpdateDualVar();
    UpdateStepSizes();
    UpdatePrimalVar();

#ifdef USE_OPENCV
    if (params_.show) {
      imshow(std::string("u"),pd_algo_.u);
      cv::waitKey(1);
    }
#endif
    if(ii_iter%100 == 0 &&
       ii_iter > 0 &&
       CheckExecutionCriterion(ii_iter))
      break;
  }//for-loop iterations

#ifdef USE_OPENCV
  if (params_.show) cv::destroyWindow("u");
#endif
  if (params_.verbose) printf("\n");
}//MumfordShah::PDAlgorithm


inline void MumfordShah::UpdateStepSizes() {
  pd_algo_.theta = 1.f / sqrt(1.f + 2.f * params_.gamma * pd_algo_.tau);
  pd_algo_.tau = pd_algo_.tau * pd_algo_.theta;
  pd_algo_.sigma = pd_algo_.sigma / pd_algo_.theta;
} //MumfordShah::UpdateStepSizes


#ifndef USE_CUDA
void CalcGradient(const float  * u, float &u_x, float &u_y,
                  size_t x, size_t y, size_t width, size_t height,
                  const bool * mask) {
  size_t i = x+y*width;

  u_x = 0.f;
  u_y = 0.f;

  if (((x+1) < width) && mask[i] && mask[i+1]) {
    u_x = u[i + 1] - u[i];
  }

  if (((y+1) < height) && mask[i] && mask[i+width]) {
    u_y = u[i + width] - u[i];
  }
}


float CalcDivergence(const float * p1, const float * p2,
                     size_t x, size_t y, size_t width, size_t pos,
                     const bool * mask) {
  size_t i_mask = x + y * width;

  float p1x = 0.f, p2y = 0.f;

  if ((x>0) && mask[i_mask] && mask[i_mask-1])
    p1x = p1[pos] - p1[pos-1];

  if ((y>0) && mask[i_mask] && mask[i_mask-width])
    p2y = p2[pos] - p2[pos-width];

  return -(p1x+p2y);
}


// compute minimizer of  0.5 * (c * x - f)^2 + (1/(2tau)) (x - x0)^2
inline float SquareProx(float x0, float c, float f, float tau) {
  return (x0 + 2.f * tau * c * f) / (1.f + 2.f * tau * c * c);
}


void MumfordShah::UpdateDualVar() {
  size_t width = pd_algo_.u.width();
  size_t height = pd_algo_.u.height();
  size_t channels = pd_algo_.u.channels();

  for (size_t y=0; y<height; y++)
    for (size_t x=0; x<width; x++) {
      if(!data_.mask.data_[x+y*width])
        continue;

      // update p variable
      float norm_p_squared = 0.f;
      for (int ch = 0; ch < channels; ch++) {
        const int idx1 = getLIdx(x,y,ch,width,height);
        const int idx2 = idx1 + pd_algo_.u.Pixels();

        float u_bar_x, u_bar_y;
        CalcGradient(&pd_algo_.u_bar.data_[ch*width*height], u_bar_x, u_bar_y,
            x, y, width, height, data_.mask.data_);

        pd_algo_.p.data_[idx1] += pd_algo_.sigma * u_bar_x;
        pd_algo_.p.data_[idx2] += pd_algo_.sigma * u_bar_y;

        norm_p_squared += pd_algo_.p.data_[idx1]*pd_algo_.p.data_[idx1];
        norm_p_squared += pd_algo_.p.data_[idx2]*pd_algo_.p.data_[idx2];
      }

      if (params_.alpha == -1.f) {
        if (norm_p_squared > 2.f * params_.lambda * pd_algo_.sigma) {
          for (int ch = 0; ch < channels; ch++) {
            const int idx1 = getLIdx(x,y,ch,width,height);
            const int idx2 = idx1 + pd_algo_.u.Pixels();
            pd_algo_.p.data_[idx1] = 0.f;
            pd_algo_.p.data_[idx2] = 0.f;
          }
        }
      } else if( params_.alpha > 0.f) {
        float expression = params_.lambda/params_.alpha *
            pd_algo_.sigma * (pd_algo_.sigma + 2.f*params_.alpha);
        if (norm_p_squared > expression) {
          for (int ch = 0; ch < channels; ch++) {
            const int idx1 = getLIdx(x,y,ch,width,height);
            const int idx2 = idx1 + pd_algo_.u.Pixels();
            pd_algo_.p.data_[idx1] = 0.f;
            pd_algo_.p.data_[idx2] = 0.f;
          }
        } else {
          float scale = 2*params_.alpha/(pd_algo_.sigma + 2*params_.alpha);
          for (int ch = 0; ch < channels; ch++) {
            const int idx1 = getLIdx(x,y,ch,width,height);
            const int idx2 = idx1 + pd_algo_.u.Pixels();
            pd_algo_.p.data_[idx1] = scale*pd_algo_.p.data_[idx1];
            pd_algo_.p.data_[idx2] = scale*pd_algo_.p.data_[idx2];
          }
        }
      }
    }
} //MumfordShah::UpdateDualVar


void MumfordShah::UpdatePrimalVar() {
  size_t width = pd_algo_.u.width();
  size_t height = pd_algo_.u.height();
  size_t channels = pd_algo_.u.channels();

  for (size_t ch=0; ch<channels; ch++)
    for (size_t y=0; y<height; y++)
      for (size_t x=0; x<width; x++) {
        if(!data_.mask.data_[x+y*width])
          continue;

        const size_t pos = getLIdx(x,y,ch,width,height);
        const float u_old = pd_algo_.u.data_[pos];
        const float divp = CalcDivergence( &pd_algo_.p.data_[0],//p1
            &pd_algo_.p.data_[pd_algo_.u.Pixels()], //p2
            x, y, width, pos, data_.mask.data_ );

        pd_algo_.u.data_[pos] = SquareProx(u_old - pd_algo_.tau * divp, // Primal descent in u (x0)
                                           data_.scalar_op.data_[pos], //c
                                           data_.intensity.data_[pos], //f
                                           pd_algo_.tau); //tau

        pd_algo_.u_diff.data_[pos] = pd_algo_.u.data_[pos] - u_old;
        pd_algo_.u_bar.data_[pos] = pd_algo_.u.data_[pos]
            + pd_algo_.theta * pd_algo_.u_diff.data_[pos];
      }
} //MumfordShah::UpdatePrimalVar
#endif //USE_CUDA


bool MumfordShah::CheckExecutionCriterion(int ii_iter) {
#ifdef USE_CUDA
  float l1norm = CUDA::norm2D(pd_algo_.u_diff.data_,
                              pd_algo_.u.width(),
                              pd_algo_.u.height()*pd_algo_.u.channels(),
                              std::string("L1"))/data_.img_domain;
#else//not USE_CUDA
  float l1norm = norm(pd_algo_.u_diff, data_.mask , "L1")/data_.img_domain;
#endif //USE_CUDA
  if (params_.verbose)
    std::cout << "iter: " << ii_iter <<
                 " done. ||u_old - u_now||_1 = " <<
                 l1norm << std::endl;

  //execution criterion
  if (l1norm < params_.tol)
    return true;
  else
    return false;

}//MumfordShah::CheckExecutionCriterion


#ifdef USE_OPENCV
bool MumfordShah::InputArgumentHandling(int argc, const char *argv[]) {
  // parse command line arguments
  const char *keys = {  // shortcut | default value | description
                        "{i       ||<none> |(path to) rgb file}"
                        "{s       ||<none> |(path to) scalar_operator file. Default: identity}"
                        "{m       ||<none> |(path to) mask file}"
                        "{o       ||<none> |output filename (will be stored in 'data')}"
                        "{p       ||<none> |absolute path to data folder. Default: path/to/mumfordShah/data/}"
                        "{l       ||0.1f   |trade-off parameter lambda. Default 0.1f}"
                        "{a       ||10.f   |alpha parameter controlling the smoothness. alpha=infinity means piecewise constant result (can be triggered with alpha=-1). Default 10}"
                        "{iter    ||5000  |number of iterations. Default 5000}"
                        "{tol     ||1e-5f  |tolerance of stopping criterion. Default 1e-5f}"
                        "{verbose ||1      |verbose output during algorithm. Default true}"
                        "{showall ||0      |show results on the fly after each iteration. Default false}"
                        "{show    ||1      |show final result. Default true}"
                        "{gamma   ||2.f    |todo}"
                        "{sigma   ||0.5f   |initial step size of dual variable}"
                        "{tau     ||0.25f  |initial step size of primal variable}"
                     };
  cv::CommandLineParser cmd(argc, argv, keys);
  std::string data            = cmd.get<std::string>("p");
  std::string img_file        = cmd.get<std::string>("i");
  std::string scalar_op_file  = cmd.get<std::string>("s");
  std::string mask_file       = cmd.get<std::string>("m");
  std::string output_prefix   = cmd.get<std::string>("o");
  float lambda                = cmd.get<float>("l");
  float alpha                 = cmd.get<float>("a");
  int max_iter                = cmd.get<int>("iter");
  float tol                   = cmd.get<float>("tol");
  bool verbose                = static_cast<bool>(cmd.get<int>("verbose"));
  bool show_all               = static_cast<bool>(cmd.get<int>("showall"));
  bool show_result            = static_cast<bool>(cmd.get<int>("show"));
  float gamma                 = cmd.get<float>("gamma");
  float sigma                 = cmd.get<float>("sigma");
  float tau                   = cmd.get<float>("tau");

  // if no image is parsed print usage of program
  if (img_file == "<none>") {
    std::cout << std::endl << "USAGE:" << std::endl;
    std::cout << "./mumfordShah -i <img.png> [-s <scalar_op.png>]";
    std::cout << "[-m <mask.png>] [-o <output.png>] [-p path/to/files/]";
    std::cout << "[-l <float>] [-a <float>] [-iter <int>] [-tol <float>]";
    std::cout << "[-verbose <bool>] [-showall <bool>] [-show <bool>]";
    std::cout << "[-gamma <float>] [-sigma <float>] [-tau <float>]";
    std::cout << std::endl << std::endl;
    std::cout << "PARAMETER DESCRIPTION:" << std::endl;
    std::cout << "shortcut | default value  | description" << std::endl;
    std::cout << "{i       |<none>          |(path to) image file}" << std::endl;
    std::cout << "{s       |<none>          |(path to) scalar_operator file. Default: identity}" << std::endl;
    std::cout << "{m       |<none>          |(path to) mask file. Default: ones matrix}" << std::endl;
    std::cout << "{o       |<none>          |output filename (will be stored in 'data'). Default: no storing}" << std::endl;
    std::cout << "{p       |<none>          |absolute path to data folder. Default: path/to/mumfordShah/data/}" << std::endl;
    std::cout << "{l       |0.1f            |trade-off parameter lambda. Default 0.1f}" << std::endl;
    std::cout << "{a       |10.f            |alpha parameter controlling the smoothness. alpha=infinity means piecewise constant result (can be triggered with alpha=-1). Default 10}" << std::endl;
    std::cout << "{iter    |5000           |number of iterations. Default 5000}" << std::endl;
    std::cout << "{tol     |1e-5f           |tolerance of stopping criterion. Default 1e-5f}" << std::endl;
    std::cout << "{verbose |1               |verbose output during algorithm. Default 1}" << std::endl;
    std::cout << "{showall |0               |show results on the fly after each iteration. Default 0}" << std::endl;
    std::cout << "{show    |1               |show final result. Default 1}" << std::endl;
    std::cout << "{gamma   |2.f             |todo}" << std::endl;
    std::cout << "{sigma   |0.5f            |initial step size of dual variable}" << std::endl;
    std::cout << "{tau     |0.25f           |initial step size of primal variable}" << std::endl;
    std::cout << "EXAMPLE:" << std::endl;
    std::cout << "./mumfordShah -i intensity.png -m mask.png -s shading.png -o result.png -a -1" << std::endl << std::endl;
    return false;
  }

  // set variables
  set_lambda(lambda);
  set_alpha(alpha);  // -1 means piecewise constant);
  set_max_iter(max_iter);
  set_tol(tol);
  set_verbose(verbose);
  set_gamma(gamma);
  set_pd_algo_sigma(sigma);
  set_pd_algo_tau(tau);
  set_show_result(show_result);
  set_show(show_all);

  // check and set data
  if (data == "<none>")
    data = std::string(MAKE_STRING(APP_SOURCE_DIR)) + "/data/";

  cv::Mat img;
  if (!LoadRgb(data + img_file, img)) {
    std::cout << "Could not load img file at '" <<
                 data + img_file << "'." << std::endl;
    return false;
  }
  data_.intensity = Image<float>(img);

  cv::Mat scalar_op;
  if (scalar_op_file == "<none>") {
    std::cout << "img.type():"<<img.type() << std::endl;
    scalar_op = cv::Mat::ones(img.rows,img.cols,img.type());
  } else if (!LoadRgb(data + scalar_op_file, scalar_op)) {
    std::cout << "Could not load scalar_op file at '" <<
                 data + scalar_op_file << "'." << std::endl;
    return false;
  }
  data_.scalar_op = Image<float>(scalar_op);

  cv::Mat mask;
  if (mask_file == "<none>") {
    mask = cv::Mat::ones(img.rows,img.cols,CV_8U);
  } else if (!LoadMask(data + mask_file, mask)) {
    std::cout << "Could not load mask file at '" <<
                 data + mask_file << "'." << std::endl;
    return false;
  }
  data_.mask = Image<bool>(mask);

  if (!(output_prefix == "<none>")) data_.output_prefix  = "";
  else data_.output_prefix = data + output_prefix;

  return true;
}
#endif // USE_OPENCV
