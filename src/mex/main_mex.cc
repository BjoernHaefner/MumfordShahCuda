/*
 * main_mex.cc
 *
 *  Created on: Oct 10, 2017
 *      Author: haefner
 */

//cpp stl
#include <string>
#include <map>
#include <functional>
#include <vector>
#include <map>
#include <sstream>
#include <memory>

// matlab headers
#include "mex.h"
#include "matrix.h"

//project dependencies
#include "../lib/mumford_shah.h"
#include "mex_utils.h"



static std::vector<std::shared_ptr<MumfordShah> > MumfordShah_;

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

static void initMumfordShah(MEX_ARGS){

  MumfordShah *ms = new MumfordShah;

  //only set variables if corresponding argument in not empty. if setting fails, return.
  if (!mxIsEmpty(prhs[0])) {
    if (!ms->set_lambda((float)mxGetScalar(prhs[0]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[1])) {
    if (!ms->set_alpha((float)mxGetScalar(prhs[1]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[2])) {
    if (!ms->set_max_iter((int)mxGetScalar(prhs[2]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[3])) {
    if (!ms->set_tol((float)mxGetScalar(prhs[3]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[4])) {
    if (!ms->set_gamma((float)mxGetScalar(prhs[4]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[5])) {
    if (!ms->set_verbose((bool)mxGetScalar(prhs[5]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[6])) {
    if (!ms->set_pd_algo_tau((float)mxGetScalar(prhs[6]))) {
      delete ms;
      return;
    }
  }
  if (!mxIsEmpty(prhs[7])) {
    if (!ms->set_pd_algo_sigma((float)mxGetScalar(prhs[7]))) {
      delete ms;
      return;
    }
  }

  MumfordShah_.push_back(std::shared_ptr<MumfordShah>(ms));
  plhs[0] = ptr_to_handle(ms);
}

static void setDataMumfordShah(MEX_ARGS) {
  MumfordShah *ms = handle_to_ptr<MumfordShah>(prhs[0]);

  float *rgb = (float*)mxGetPr(prhs[1]);
  bool *mask = mxGetLogicals(prhs[2]);

  const mwSize *dims = mxGetDimensions(prhs[1]);
  int rows = (int)dims[0];
  int cols = (int)dims[1];
  int channels = (int)dims[2];

  ms->SetData(rgb, mask, rows, cols, channels, std::string("matlab"));
}

static void runMumfordShah(MEX_ARGS){
  MumfordShah *ms = handle_to_ptr<MumfordShah>(prhs[0]);

  float *scalar_op = (float*)mxGetPr(prhs[1]);

  plhs[0] = mxCreateNumericArray(3, mxGetDimensions(prhs[1]),
      mxSINGLE_CLASS, mxREAL);

  float *result =(float*)mxGetPr(plhs[0]);
  ms->Run(scalar_op, result, std::string("matlab"));
}

static void closeMumfordShah(MEX_ARGS){
  MumfordShah *ms = handle_to_ptr<MumfordShah>(prhs[0]);
  delete ms;
}

const static std::map<std::string, std::function<void(MEX_ARGS)>> cmd_reg = {

{ "initMumfordShah", initMumfordShah },
{ "setDataMumfordShah", setDataMumfordShah },
{ "runMumfordShah", runMumfordShah },
{ "closeMumfordShah", closeMumfordShah },

    };

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  if(nrhs == 0)
    mexErrMsgTxt("Usage: mumfordShahMEX(command, arg1, arg2, ...);");

  char *cmd = mxArrayToString(prhs[0]);
  bool executed = false;

  for(auto& c : cmd_reg)
  {
    if(c.first.compare(cmd) == 0)
    {
      c.second(nlhs, plhs, nrhs - 1, prhs + 1);
      executed = true;
      break;
    }
  }

  if(!executed)
  {
    std::stringstream msg;
    msg << "Unknown command '" << cmd << "'. List of supported commands:";
    for(auto& c : cmd_reg)
      msg << "\n - " << c.first.c_str();

    mexErrMsgTxt(msg.str().c_str());
  }

  mexEvalString("drawnow;");

  return;
}
