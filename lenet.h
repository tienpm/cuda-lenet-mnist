#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <unistd.h>

const size_t ALL_IN_ONE_BIN_BYTES = 55196824;

void initialize(void* all_in_one_buffer, bool mode_training, int max_batch_size);
void finalize(bool mode_training);

void train(int num_training_items, int num_training_batch, float learning_rate);
void inference(int num_inference_items, int num_inference_batch);

// util functions
void uint8_to_float(uint8_t* in, float* out, int N);
int check_answer(float* pred, uint8_t* label, int N, int C);

// forward functions
void convolution(float* in, float* weight, float* bias, float* out, int N, int C, int H, int W, int K, int R, int S, int pad);
void sigmoid_inplace(float* inout, int N);
void maxpool(float* in, float* out, int N, int C, int H, int W, int R, int S, int stride);
void linear(float* in, float* weight, float* bias, float* output, int N, int C, int K);
float cross_entropy_loss(float* pred, uint8_t* label, float* softmax_buf, int N, int C);

// backward functions
void convolution_backward_input(float* out_grad, float* weight, float* in_grad, int N, int C, int H, int W, int K, int R, int S, int pad);
void convolution_backward_weight(float* out_grad, float* in, float* weight_grad, int N, int C, int H, int W, int K, int R, int S, int pad);
void convolution_backward_bias(float* out_grad, float* bias_grad, int N, int C, int H, int W, int K, int R, int S, int pad);
void sigmoid_backward_inplace(float* inout, float* inout_grad, int N);
void maxpool_backward(float* out_grad, float* in, float* in_grad, int N, int C, int H, int W, int R, int S, int stride);
void linear_backward_input(float* weight, float* out_grad, float* in_grad, int N, int C, int K);
void linear_backward_weight(float* in, float* out_grad, float* weight_grad, int N, int C, int K);
void linear_backward_bias(float* out_grad, float* bias_grad, int N, int K);
void cross_entropy_loss_backward(float* softmax_buf, uint8_t* label, float* in_grad, int N, int C);

// parameter update functions
void sgd(float* weight, float* grad, float lr, int N);