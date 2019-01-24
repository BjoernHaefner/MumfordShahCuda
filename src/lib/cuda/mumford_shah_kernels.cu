//CUDA
#include "../mumford_shah.h"
#include "cuda_common.cuh"
#include "cuda_kernels.cuh"


__device__ void d_calcGradient(const float *u,
                               float &u_x,
                               float &u_y,
                               size_t width,
                               size_t height,
                               const bool * mask) {
  // Get the 2D-coordinate of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;
  // Get linear address of current coordinate in the image
  size_t i = x+y*width;

  u_x = 0.f;
  u_y = 0.f;
  if ((x+1)<width && mask[i] && mask[i+1]) u_x = u[i+1] - u[i];
  if ((y+1)<height && mask[i] && mask[i+width]) u_y = u[i+width] - u[i];
}

__global__ void g_updateDualVars(float *p,
                                 float sigma,
                                 const float *u_bar,
                                 float alpha, float lambda,
                                 size_t width, size_t height, size_t channels,
                                 const bool *mask) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x>=width) || (y>=height)) return;
  if (!mask[x+y*width]) return;

  const int N = width*height;
  // update p variable
  float norm_p_squared = 0.f;
  for(int ch = 0; ch < channels; ch++) {
    const int idx1 = CUDA::getLIdx(x,y,ch,width,height);
    const int idx2 = idx1 + N*channels;
    
    float u_bar_x, u_bar_y;
    d_calcGradient(&u_bar[ch*N], u_bar_x, u_bar_y, width, height, mask);

    p[idx1] += sigma * u_bar_x;
    p[idx2] += sigma * u_bar_y;

    norm_p_squared += p[idx1]*p[idx1];
    norm_p_squared += p[idx2]*p[idx2];
  }

  if (alpha == -1.f) {
    if(norm_p_squared > 2.f * lambda * sigma) {
      for (int ch = 0; ch < channels; ch++) {
        const int idx1 = CUDA::getLIdx(x,y,ch,width,height);;
        const int idx2 = idx1 + N*channels;
        p[idx1] = 0.f;
        p[idx2] = 0.f;
      }
    }
  } else if(alpha > 0.f) {
    float expression = lambda/alpha * sigma * (sigma + 2.f * alpha);
    if (norm_p_squared > expression) {
      for (int ch = 0; ch < channels; ch++) {
        const int idx1 = CUDA::getLIdx(x,y,ch,width,height);
        const int idx2 = idx1 + N*channels;
        p[idx1] = 0.f;
        p[idx2] = 0.f;
      }
    } else {
      float scale = 2*alpha/(sigma + 2*alpha);
      for (int ch = 0; ch < channels; ch++) {
        const int idx1 = CUDA::getLIdx(x,y,ch,width,height);
        const int idx2 = idx1 + N*channels;
        p[idx1] = scale*p[idx1];
        p[idx2] = scale*p[idx2];
      }
    }
  }
}



//#######################################################################
void MumfordShah::UpdateDualVar() {
  g_updateDualVars<<< cuda_params_.dimGrid2D, cuda_params_.dimBlock2D >>>(
      pd_algo_.p.data_,
      pd_algo_.sigma,
      pd_algo_.u_bar.data_,
      params_.alpha, params_.lambda,
      pd_algo_.u.width(), pd_algo_.u.height(), pd_algo_.u.channels(),
      data_.mask.data_);
  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL(cudaPeekAtLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


__device__ void d_calcDivergence(const float *v1,
                                 const float *v2,
                                 float &divv,
                                 size_t width, size_t height, size_t c,
                                 const bool *mask) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t i_mask = x + y * width;
  size_t i      = x + y*width + c * width * height;

  float v1x = 0.f, v2y = 0.f;
  if (x>0 && mask[i_mask] && mask[i_mask-1]) v1x = v1[i] - v1[i-1];
  if (y>0 && mask[i_mask] && mask[i_mask-width]) v2y = v2[i] - v2[i-width];
  divv = -( v1x + v2y );
}


// compute minimizer of  0.5 * (c * x - f)^2 + (1/(2tau)) (x - x0)^2
__device__ inline float d_square_prox(float x0, float c, float f, float tau) {
  return (x0 + 2.f * tau * c * f) / (1.f + 2.f * tau * c * c);
}

//############################################################################
__global__ void g_updatePrimalVar(float *u, float *u_bar, float *u_diff,
                                  const float *p, const float *f,
                                  const float *scalar_op,
                                  float tau, float theta,
                                  size_t width, size_t height, size_t channels,
                                  const bool *mask) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x>=width || y>=height) return;
  if(!mask[x+y*width]) return;

  for(int c = 0; c < channels; c++) {
    const size_t i = x + y * width + c * width * height;
    const float u_old = u[i];

    float divp;
    d_calcDivergence( &p[0], &p[width*height*channels], divp, width, height, c, mask );

    const float u_new = d_square_prox(u_old - tau * divp, scalar_op[i], f[i], tau);
    u_bar[i] = u_new + theta * (u_new - u_old);
    u[i] = u_new;
    u_diff[i] = abs(u_new - u_old);
  }
}



//############################################################################
void MumfordShah::UpdatePrimalVar() {
  g_updatePrimalVar<<< cuda_params_.dimGrid2D, cuda_params_.dimBlock2D >>>(
      pd_algo_.u.data_,
      pd_algo_.u_bar.data_,
      pd_algo_.u_diff.data_,
      pd_algo_.p.data_,
      data_.intensity.data_,
      data_.scalar_op.data_,
      pd_algo_.tau,
      pd_algo_.theta,
      pd_algo_.u.width(), pd_algo_.u.height(), pd_algo_.u.channels(),
      data_.mask.data_);
  // check if errors occured and wait until results are ready
  CUDA_SAFE_CALL(cudaPeekAtLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
