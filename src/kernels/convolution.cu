#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include "convolution.cuh"
#include "util.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_MPI(call)                                                        \
  do {                                                                         \
    int code = call;                                                           \
    if (code != MPI_SUCCESS) {                                                 \
      char estr[MPI_MAX_ERROR_STRING];                                         \
      int elen;                                                                \
      MPI_Error_string(code, estr, &elen);                                     \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr);    \
      MPI_Abort(MPI_COMM_WORLD, code);                                         \
    }                                                                          \
  } while (0)

#define WARP_SIZE 16
#define NGPUS 4

static float *I_gpu;
static float *Col_gpu;
static float *F_gpu;
static float *O_gpu;
static float *I_gpus[NGPUS];
static float *Col_gpus[NGPUS];
static float *F_gpus[NGPUS];
static float *O_gpus[NGPUS];
static int mpi_rank, mpi_world_size;

__global__ void gpu_im2col(float *_I, float *workspace, int N, int C, int H,
                           int W, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w);

__global__ void gpu_matmul(float *A, float *B, float *C, int M, int K, int N);

__global__ void gpu_matmul_tc(float *A, float *B, float *C, int M, int N,
                              int K);

__global__ void gpu_reshape(float *_src, float *_dst, int N, int K, int OH,
                            int OW);

void im2col_gpu_convolution(float *_I, float *_F, float *_O, float *_BUF1,
                            float *_BUF2, int N, int C, int H, int W, int K,
                            int R, int S, int pad_h, int pad_w, int stride_h,
                            int stride_w, int dilation_h, int dilation_w);

void im2col_gpu_convolution_thread(float *_I, float *_F, float *_O,
                                   float *I_gpus, float *Col_gpus,
                                   float *F_gpus, float *O_gpus, size_t *Nbegin,
                                   size_t *Nend, int N, int C, int H, int W,
                                   int K, int R, int S, int pad_h, int pad_w,
                                   int stride_h, int stride_w, int dilation_h,
                                   int dilation_w, int gpu_id);

void im2col_convolution_multi_gpu(float *_I, float *_F, float *_O, int N, int C,
                                  int H, int W, int K, int R, int S, int pad_h,
                                  int pad_w, int stride_h, int stride_w,
                                  int dilation_h, int dilation_w);

void convolution(float *_I, float *_F, float *_O, float *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w, int _mpi_rank, int _mpi_world_size) {
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  const int FILTER = C * R * S;
  const int OUTPUT = OH * OW;
  mpi_rank = _mpi_rank;
  mpi_world_size = _mpi_world_size;
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution_im2col(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
  //                              pad_h, pad_w, stride_h, stride_w, dilation_h,
  //                              dilation_w);

  /* ===== Convolution im2col on GPU  =====*/
  // im2col_gpu_convolution(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
  // pad_h,
  //                        pad_w, stride_h, stride_w, dilation_h, dilation_w);

  /* ===== Convolution im2col on GPU  =====*/
  // im2col_convolution_multi_gpu(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w,
  //                              stride_h, stride_w, dilation_h, dilation_w);

  /* ===== Convolution im2col on GPU MPI =====*/
  int nrows_per_proc = N / mpi_world_size;
  int main_worker = 0;

  MPI_Request bcast_request;
  // Scatter matrix I to nodes with chunk of data and broadcast matrix F
  CHECK_MPI(MPI_Ibcast(_F, K * C * R * S, MPI_FLOAT, main_worker,
                       MPI_COMM_WORLD, &bcast_request));
  CHECK_MPI(MPI_Scatter(_I, nrows_per_proc * C * H * W, MPI_FLOAT, _I,
                        nrows_per_proc * C * H * W, MPI_FLOAT, main_worker,
                        MPI_COMM_WORLD));
  MPI_Wait(&bcast_request, MPI_SUCCESS);

  // Calculate im2col
  // printf("Calculate begin on rank %d\n", mpi_rank);
  im2col_convolution_multi_gpu(_I, _F, _O, nrows_per_proc, C, H, W, K, R, S,
                               pad_h, pad_w, stride_h, stride_w, dilation_h,
                               dilation_w);
  // printf("Calculate finished on rank %d\n", mpi_rank);

  CHECK_MPI(MPI_Gather(_O, nrows_per_proc * K * OUTPUT, MPI_FLOAT, _O,
                       nrows_per_proc * K * OUTPUT, MPI_FLOAT, main_worker,
                       MPI_COMM_WORLD));
}

void convolution_init(int N, int C, int H, int W, int K, int R, int S,
                      int pad_h, int pad_w, int stride_h, int stride_w,
                      int dilation_h, int dilation_w) {
  /* ===== Convolution im2col on GPU  =====*/
  // const int OH = (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h +
  // 1; const int OW = (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w
  // + 1; const int FILTER = C * R * S; const int OUTPUT = OH * OW;
  // CHECK_CUDA(cudaMalloc((void **)&I_gpu, N * C * H * W * sizeof(float)));
  // CHECK_CUDA(
  //     cudaMalloc((void **)&Col_gpu,
  //                N * FILTER * OUTPUT * sizeof(float))); // N * FILTER *
  //                OUTPUT
  // CHECK_CUDA(
  //     cudaMalloc((void **)&F_gpu, K * FILTER * sizeof(float))); // K * FILTER
  // CHECK_CUDA(cudaMalloc((void **)&O_gpu,
  //                       N * K * OUTPUT *
  //                           sizeof(float))); // (K * FILTER) * (N * FILTER *
  //                                            // OUTPUT) = (N * K * OUTPUT)
  //
  // CHECK_CUDA(cudaDeviceSynchronize());

  /* ===== Convolution im2col on GPU  =====*/
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  const int FILTER = C * R * S;
  const int OUTPUT = OH * OW;
  size_t Nbegin[NGPUS], Nend[NGPUS];
  for (size_t i = 0; i < NGPUS; i++) {
    Nbegin[i] = N / NGPUS * i;
    Nend[i] = N / NGPUS * (i + 1);
    if (i == NGPUS - 1)
      Nend[i] = N;
  }

  for (size_t i = 0; i < NGPUS; i++) {
    cudaSetDevice(i);
    CHECK_CUDA(cudaMalloc((void **)&I_gpus[i],
                          (Nend[i] - Nbegin[i]) * C * H * W * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc((void **)&Col_gpus[i],
                   (Nend[i] - Nbegin[i]) * FILTER * OUTPUT * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&F_gpus[i], K * FILTER * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&O_gpus[i],
                          (Nend[i] - Nbegin[i]) * K * OUTPUT * sizeof(float)));

    // Memcopy from pagable to pinned memory
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  /* ===== Convolution im2col on GPU  =====*/
  // CHECK_CUDA(cudaFree(I_gpu));
  // CHECK_CUDA(cudaFree(Col_gpu));
  // CHECK_CUDA(cudaFree(F_gpu));
  // CHECK_CUDA(cudaFree(O_gpu));
  //
  // CHECK_CUDA(cudaDeviceSynchronize());

  /* ===== Convolution im2col on GPU  =====*/
  for (int i = 0; i < NGPUS; i++) {
    cudaSetDevice(i);
    CHECK_CUDA(cudaFree(I_gpus[i]));
    CHECK_CUDA(cudaFree(Col_gpus[i]));
    CHECK_CUDA(cudaFree(F_gpus[i]));
    CHECK_CUDA(cudaFree(O_gpus[i]));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

__global__ void gpu_im2col(float *_I, float *workspace, int N, int C, int H,
                           int W, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = _I;

  // GPU im2col
  const int OH = (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h + 1;
  const int OW = (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w + 1;
  const int FILTER = C * R * S;
  const int BATCH_OFFSET = FILTER * OH * OW;

  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  // if (batch > FILTER)
  //   return;
  int out = blockIdx.x * blockDim.x + threadIdx.x;

  if (out < OH * OW) {
    int y = stride_h * (out / OW) - pad_h;
    int x = stride_w * (out % OW) - pad_w;
    int idx = 0;
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < R; h++) {
        for (int w = 0; w < S; w++) {
          int dy = y + h * dilation_h;
          int dx = x + w * dilation_w;
          if ((0 <= dy && dy < H) && (0 <= dx && dx < W)) {
            workspace[batch * BATCH_OFFSET + (idx * OH * OW) + out] =
                I[batch * (C * H * W) + c * (H * W) + dy * W + dx];
          } else {
            workspace[batch * BATCH_OFFSET + (idx * OH * OW) + out] = 0;
          }
          idx++;
        }
      }
    }
  }
}

__global__ void gpu_matmul(float *A, float *B, float *C, int M, int K, int N) {
  __shared__ float sA[WARP_SIZE][WARP_SIZE + 1];
  __shared__ float sB[WARP_SIZE][WARP_SIZE];

  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int sy = threadIdx.y;
  int sx = threadIdx.x;

  float sum = 0;
  for (int t = 0; t < K; t += WARP_SIZE) {
    // A
    if (y < M && (sx + t) < K) {
      sA[sy][sx] = A[y * K + (sx + t)];
    } else {
      sA[sy][sx] = 0;
    }
    // B
    if (x < N && (sy + t) < K) {
      sB[sy][sx] = B[batch * K * N + (sy + t) * N + x];
    } else {
      sB[sy][sx] = 0;
    }
    // sync
    __syncthreads();
    // matmul
    for (int k = 0; k < WARP_SIZE && k + t < K; k++) {
      sum += sA[sy][k] * sB[k][sx];
    }
    __syncthreads();
  }
  if (y < M && x < N) {
    C[batch * M * N + y * N + x] = sum;
  }
}

__global__ void gpu_reshape(float *_src, float *_dst, int N, int K, int OH,
                            int OW) {}

void im2col_gpu_convolution_thread(float *_I, float *_F, float *_O,
                                   float **I_gpus, float **Col_gpus,
                                   float **F_gpus, float **O_gpus,
                                   size_t *Nbegin, size_t *Nend, int N, int C,
                                   int H, int W, int K, int R, int S, int pad_h,
                                   int pad_w, int stride_h, int stride_w,
                                   int dilation_h, int dilation_w, int gpu_id) {
  float *I = _I, *F = _F, *O = _O;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  const int FILTER = C * R * S;
  const int OUTPUT = OH * OW;
  cudaSetDevice(gpu_id);
  const int num_rows = Nend[gpu_id] - Nbegin[gpu_id];

  // Move I, F to GPU
  CHECK_CUDA(cudaMemcpy(I_gpus[gpu_id], &I[Nbegin[gpu_id] * C * H * W],
                        num_rows * C * H * W * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_gpus[gpu_id], F, K * FILTER * sizeof(float),
                        cudaMemcpyHostToDevice));

  // im2col
  dim3 blockDim_im2col(WARP_SIZE * WARP_SIZE, 1);
  dim3 gridDim_im2col((OUTPUT + blockDim_im2col.x - 1) / blockDim_im2col.x,
                      num_rows);
  gpu_im2col<<<gridDim_im2col, blockDim_im2col>>>(
      I_gpus[gpu_id], Col_gpus[gpu_id], num_rows, C, H, W, R, S, pad_h, pad_w,
      stride_h, stride_w, dilation_h, dilation_w);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // matmul
  dim3 blockDim_matmul(WARP_SIZE, WARP_SIZE, 1);
  dim3 gridDim_matmul((OUTPUT + blockDim_matmul.x - 1) / blockDim_matmul.x,
                      (K + blockDim_matmul.y - 1) / blockDim_matmul.y,
                      num_rows);
  gpu_matmul<<<gridDim_matmul, blockDim_matmul>>>(
      F_gpus[gpu_id], Col_gpus[gpu_id], O_gpus[gpu_id], K, FILTER, OUTPUT);
  // CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  // reshape
  CHECK_CUDA(cudaMemcpy(&O[Nbegin[gpu_id] * K * OUTPUT], O_gpus[gpu_id],
                        num_rows * K * OUTPUT * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  // printf("\nEND DEBUG - %d\n", gpu_id);
}

void im2col_convolution_multi_gpu(float *_I, float *_F, float *_O, int N, int C,
                                  int H, int W, int K, int R, int S, int pad_h,
                                  int pad_w, int stride_h, int stride_w,
                                  int dilation_h, int dilation_w) {
  float *I = _I, *F = _F, *O = _O;
  // const int OH = (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h +
  // 1; const int OW = (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w
  // + 1; const int FILTER = C * R * S; const int OUTPUT = OH * OW;

  size_t Nbegin[NGPUS], Nend[NGPUS];
  for (size_t i = 0; i < NGPUS; i++) {
    Nbegin[i] = N / NGPUS * i;
    Nend[i] = N / NGPUS * (i + 1);
    if (i == NGPUS - 1)
      Nend[i] = N;
  }

  /* 0. Allocate GPU memory */
  // cudaStream_t streams[NGPUS];
  // for (int i = 0; i < NGPUS; i++) {
  //   cudaSetDevice(i);
  //   cudaStreamCreate(&streams[i]);
  // }

  /* Launch worker threads and wait for all thread finish */

#pragma omp parallel for num_threads(NGPUS)
  for (int i = 0; i < NGPUS; i++) {
    int tid = omp_get_thread_num();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(tid, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    im2col_gpu_convolution_thread(
        I, F, O, I_gpus, Col_gpus, F_gpus, O_gpus, Nbegin, Nend, N, C, H, W, K,
        R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, i);
  }
}

// void im2col_gpu_convolution(float *_I, float *_F, float *_O, float *_BUF1,
//                             float *_BUF2, int N, int C, int H, int W, int K,
//                             int R, int S, int pad_h, int pad_w, int stride_h,
//                             int stride_w, int dilation_h, int dilation_w) {
//   float *I = _I, *F = _F, *O = _O;
//   const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) /
//   stride_h; const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1))
//   / stride_w; const int FILTER = C * R * S; const int OUTPUT = OH * OW;
//   // Move I, F to GPU
//   CHECK_CUDA(cudaMemcpy(I_gpu, I, N * C * H * W * sizeof(float),
//                         cudaMemcpyHostToDevice));
//   CHECK_CUDA(
//       cudaMemcpy(F_gpu, F, K * FILTER * sizeof(float),
//       cudaMemcpyHostToDevice));
//
//   // im2col
//   dim3 blockDim_im2col(WARP_SIZE * WARP_SIZE, 1);
//   dim3 gridDim_im2col((OUTPUT + blockDim_im2col.x - 1) / blockDim_im2col.x,
//   N); gpu_im2col<<<gridDim_im2col, blockDim_im2col>>>(
//       I_gpu, Col_gpu, N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w,
//       dilation_h, dilation_w);
//   CHECK_CUDA(cudaGetLastError());
//   CHECK_CUDA(cudaDeviceSynchronize());
//   // matmul
//   dim3 blockDim_matmul(WARP_SIZE, WARP_SIZE, 1);
//   dim3 gridDim_matmul((OUTPUT + blockDim_matmul.x - 1) / blockDim_matmul.x,
//                       (K + blockDim_matmul.y - 1) / blockDim_matmul.y, N);
//   gpu_matmul<<<gridDim_matmul, blockDim_matmul>>>(F_gpu, Col_gpu, O_gpu, K,
//                                                   FILTER, OUTPUT);
//   CHECK_CUDA(cudaGetLastError());
//   CHECK_CUDA(cudaDeviceSynchronize());
//
//   // reshape
//   CHECK_CUDA(cudaMemcpy(O, O_gpu, N * K * OUTPUT * sizeof(float),
//                         cudaMemcpyDeviceToHost));
//   // gpu_reshape<<<gridDim, blockDim>>>(BUF2, O, N, K, OH, OW);
//   // CHECK_CUDA(cudaGetLastError());
//   CHECK_CUDA(cudaDeviceSynchronize());
// }
