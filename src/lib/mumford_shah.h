
#ifndef MUMFORDSHAH_LIB_MUMFORD_SHAH_H_
#define MUMFORDSHAH_LIB_MUMFORD_SHAH_H_

//to use cmake defined variable APP_SOURCE_DIR
#define MAKE_STRING_(x)  #x
#define MAKE_STRING(x)  MAKE_STRING_(x)

//STL
#include <string>

//opencv
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

//local code
#include "image.h"

//#ifndef DEBUG
//#define DEBUG
//#endif

class MumfordShah
{
  //##########################################################################
  //################################PUBLIC####################################
  //##########################################################################

public:

  MumfordShah();
#ifdef USE_OPENCV
  MumfordShah(int argc, const char *argv[]);
#endif
  ~MumfordShah();

  //the following three 'Run' functions are intended to be used, if MumfordShah shall run once
#ifdef USE_OPENCV
  bool Run(float *result = NULL);
  bool Run(const std::string &rgb, const std::string &scalar_op,
           const std::string &mask,const std::string &output_file);
  bool Run(const cv::Mat &rgb, const cv::Mat &scalar_op, const cv::Mat &mask,
           cv::Mat &result);
#endif
  bool Run(const float *rgb, const float *scalar_op, const bool *mask,
           const int rows, const int cols, const int channels,
           float *result, const std::string aligned = "row-major" );

  //'SetData' and the followup function 'Run' are intended to be used if multiple runs of mumfordshah with changing 'scalar_op' variable are of interest.
  bool SetData(const float *rgb, const bool  *mask,
               const int rows, const int cols, const int channels,
               const std::string aligned = "matlab");
  bool Run(const float *scalar_op, float *result,
           const std::string aligned = "matlab");


  bool set_lambda(float lambda) {
    if (lambda > 0.f) {
      this->params_.lambda = lambda;
      return true;
    } else {
      return false;
    }
  }
  bool set_alpha(float alpha) {
    if (alpha >= 0.f || alpha == -1.f) {
      this->params_.alpha = alpha;
      return true;
    } else {
      return false;
    }
  }
  bool set_max_iter(float max_iter) {
    if (max_iter > 0) {
      this->params_.max_iter = max_iter;
      return true;
    } else {
      return false;
    }
  }
  bool set_tol(float tol) {
    this->params_.tol = tol;
    return true;
  }
  bool set_verbose(bool verbose) {
    if (verbose == false || verbose == true) {
      this->params_.verbose = verbose;
      return true;
    } else {
      return false;
    }
  }
  bool set_gamma(float gamma) {
    if (gamma > 0.f) {
      this->params_.gamma = gamma;
      return true;
    } else {
      return false;
    }
  }
#ifdef USE_OPENCV
  bool set_show_result(bool show_result) {
    if (show_result == false || show_result == true) {
      this->params_.show_result = show_result;
      return true;
    } else {
      return false;
    }
  }
  bool set_show(bool show) {
    if (show == false || show == true) {
      this->params_.show = show;
      return true;
    } else {
      return false;
    }
  }
#endif
  bool set_pd_algo_tau(float tau) {
    if (tau > 0.f) {
      this->pd_algo_.tau_init = tau;
      return true;
    } else {
      return false;
    }
  }
  bool set_pd_algo_sigma(float sigma) {
    if (sigma > 0.f) {
      this->pd_algo_.sigma_init = sigma;
      return true;
    } else {
      return false;
    }
  }

  //##########################################################################
  //################################PRIVATE###################################
  //##########################################################################

private:

  void PrintDetails() const;
  void InitMumfordShah();
  void InitPDAlgorithm();
  void PDAlgorithm();
  void UpdateDualVar();
  void UpdatePrimalVar();
  void UpdateStepSizes();
  bool CheckExecutionCriterion(int ii_iter);
  bool InputArgumentHandling(int argc, const char *argv[]);
#ifdef DEBUG
  void Show();
  void Write();
  void Debug(const std::string &message = "");
#endif

  struct {
    int max_iter    = 5000;
    float tol       = 1e-5f; //tolerance for the (relative) primal-dual gap to execute. Relative means (E_primal-E_dual)/E_primal
    float lambda    = 0.1f; //Penalizes the length of the discontinuity set
    float alpha     = 10.f; //Penalizes smoothness of minimizer outside of discontinuities of primal variable. alpha=-1 means alpha \to\infty
    bool verbose    = true;
    float gamma     = 2.f;
#ifdef USE_OPENCV
    bool show_result = true;
    bool show = false;
#endif
  } params_;

  struct {
    Image<float> intensity, scalar_op;
    Image<bool>  mask;
    Image<float> initial_guess;
    float img_domain;
    std::string output_prefix;
  } data_;

  struct {
    Image<float> p, u, u_bar, u_diff;
    float tau_init = 0.25f;
    float sigma_init = 0.5f;
    float tau, sigma, theta;
  } pd_algo_;

#ifdef USE_CUDA
  struct {
    dim3 dimGrid3D, dimBlock3D;
    dim3 dimGrid2D, dimBlock2D;
  } cuda_params_;
#endif

};//class MumfordShah
#endif /*MUMFORDSHAH_LIB_MUMFORD_SHAH_H_*/
