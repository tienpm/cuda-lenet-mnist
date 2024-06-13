#include "lenet.h"
#include <sys/time.h>

// model parameters
static float* conv1_weight;
static float* conv1_bias;
static float* conv2_weight;
static float* conv2_bias;
static float* fc1_weight;
static float* fc1_bias;
static float* fc2_weight;
static float* fc2_bias;
static float* fc3_weight;
static float* fc3_bias;

// MNIST dataset
static uint8_t* train_images;
static uint8_t* train_labels;
static uint8_t* test_images;
static uint8_t* test_labels;

// buffers for inference
static float *act_input;
static float *act_conv1;
static float *act_pool1;
static float *act_conv2;
static float *act_pool2;
static float *act_fc1;
static float *act_fc2;
static float *act_fc3;

// buffers for training
static float *act_celoss;
static float *grad_celoss;
static float *grad_fc3_input;
static float *grad_fc3_weight;
static float *grad_fc3_bias;
static float *grad_fc2_input;
static float *grad_fc2_weight;
static float *grad_fc2_bias;
static float *grad_fc1_input;
static float *grad_fc1_weight;
static float *grad_fc1_bias;
static float *grad_pool2;
static float *grad_conv2_input;
static float *grad_conv2_weight;
static float *grad_conv2_bias;
static float *grad_pool1;
static float *grad_conv1_weight;
static float *grad_conv1_bias;

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void initialize(void* all_in_one_buffer, bool mode_training, int max_batch_size) {
  void* buffer = all_in_one_buffer;
  conv1_weight = (float*)buffer; buffer = (float*)buffer + 6 * 1 * 5 * 5; 
  conv1_bias = (float*)buffer; buffer = (float*)buffer + 6; 
  conv2_weight = (float*)buffer; buffer = (float*)buffer + 16 * 6 * 5 * 5; 
  conv2_bias = (float*)buffer; buffer = (float*)buffer + 16; 
  fc1_weight = (float*)buffer; buffer = (float*)buffer + 120 * 400; 
  fc1_bias = (float*)buffer; buffer = (float*)buffer + 120; 
  fc2_weight = (float*)buffer; buffer = (float*)buffer + 84 * 120; 
  fc2_bias = (float*)buffer; buffer = (float*)buffer + 84; 
  fc3_weight = (float*)buffer; buffer = (float*)buffer + 10 * 84; 
  fc3_bias = (float*)buffer; buffer = (float*)buffer + 10; 
  train_images = (uint8_t*)buffer; buffer = (uint8_t*)buffer + 60000 * 28 * 28;
  train_labels = (uint8_t*)buffer; buffer = (uint8_t*)buffer + 60000; 
  test_images = (uint8_t*)buffer; buffer = (uint8_t*)buffer + 10000 * 28 * 28;
  test_labels = (uint8_t*)buffer; buffer = (uint8_t*)buffer + 10000;
  if (buffer != (uint8_t*)all_in_one_buffer + ALL_IN_ONE_BIN_BYTES) {
    printf("buffer != all_in_one_buffer + ALL_IN_ONE_BIN_BYTES\n");
    exit(1);
  }

  // Prepare buffers for inference
  int N = max_batch_size;
  act_input = (float*)malloc(N * 1 * 28 * 28 * sizeof(float));
  act_conv1 = (float*)malloc(N * 6 * 28 * 28 * sizeof(float));
  act_pool1 = (float*)malloc(N * 6 * 14 * 14 * sizeof(float));
  act_conv2 = (float*)malloc(N * 16 * 10 * 10 * sizeof(float));
  act_pool2 = (float*)malloc(N * 16 * 5 * 5 * sizeof(float));
  act_fc1 = (float*)malloc(N * 120 * sizeof(float));
  act_fc2 = (float*)malloc(N * 84 * sizeof(float));
  act_fc3 = (float*)malloc(N * 10 * sizeof(float));

  // Addition buffers for training
  if (mode_training) {
    act_celoss = (float*)malloc(N * 10 * sizeof(float));
    grad_celoss = (float*)malloc(N * 10 * sizeof(float));
    grad_fc3_input = (float*)malloc(N * 84 * sizeof(float));
    grad_fc3_weight = (float*)malloc(10 * 84 * sizeof(float));
    grad_fc3_bias = (float*)malloc(10 * sizeof(float));
    grad_fc2_input = (float*)malloc(N * 120 * sizeof(float));
    grad_fc2_weight = (float*)malloc(84 * 120 * sizeof(float));
    grad_fc2_bias = (float*)malloc(84 * sizeof(float));
    grad_fc1_input = (float*)malloc(N * 400 * sizeof(float));
    grad_fc1_weight = (float*)malloc(120 * 400 * sizeof(float));
    grad_fc1_bias = (float*)malloc(120 * sizeof(float));
    grad_pool2 = (float*)malloc(N * 16 * 10 * 10 * sizeof(float));
    grad_conv2_input = (float*)malloc(N * 6 * 14 * 14 * sizeof(float));
    grad_conv2_weight = (float*)malloc(16 * 6 * 5 * 5 * sizeof(float));
    grad_conv2_bias = (float*)malloc(16 * sizeof(float));
    grad_pool1 = (float*)malloc(N * 6 * 28 * 28 * sizeof(float));
    grad_conv1_weight = (float*)malloc(6 * 1 * 5 * 5 * sizeof(float));
    grad_conv1_bias = (float*)malloc(6 * sizeof(float));
  }
}

void finalize(bool mode_training) {
  // inference buffer cleanup
  free(act_input);
  free(act_conv1);
  free(act_pool1);
  free(act_conv2);
  free(act_pool2);
  free(act_fc1);
  free(act_fc2);
  free(act_fc3);

  if (mode_training) {
    free(act_celoss);
    free(grad_celoss);
    free(grad_fc3_input);
    free(grad_fc3_weight);
    free(grad_fc3_bias);
    free(grad_fc2_input);
    free(grad_fc2_weight);
    free(grad_fc2_bias);
    free(grad_fc1_input);
    free(grad_fc1_weight);
    free(grad_fc1_bias);
    free(grad_pool2);
    free(grad_conv2_input);
    free(grad_conv2_weight);
    free(grad_conv2_bias);
    free(grad_pool1);
    free(grad_conv1_weight);
    free(grad_conv1_bias);
  }
}

void train(int num_training_items, int num_training_batch, float learning_rate) {
  int N = num_training_batch;

  printf("Training...\n");
  int progress_10th = 0;
  double start_time = get_time();
  for (int i = 0; i < num_training_items; i += N) {
    if (i + N > num_training_items) {
      N = num_training_items - i;
    }
    if (i >= num_training_items / 10 * (progress_10th + 1)) {
      ++progress_10th;
      printf("Training progress: %d / %d\n", i , num_training_items);
    }

    // forward pass
    uint8_to_float(train_images + i * 28 * 28, act_input, N * 28 * 28);
    convolution(act_input, conv1_weight, conv1_bias, act_conv1, N, 1, 28, 28, 6, 5, 5, 2);
    sigmoid_inplace(act_conv1, N * 6 * 28 * 28);
    maxpool(act_conv1, act_pool1, N, 6, 28, 28, 2, 2, 2);
    convolution(act_pool1, conv2_weight, conv2_bias, act_conv2, N, 6, 14, 14, 16, 5, 5, 0);
    sigmoid_inplace(act_conv2, N * 16 * 10 * 10);
    maxpool(act_conv2, act_pool2, N, 16, 10, 10, 2, 2, 2);
    linear(act_pool2, fc1_weight, fc1_bias, act_fc1, N, 16 * 5 * 5, 120);
    sigmoid_inplace(act_fc1, N * 120);
    linear(act_fc1, fc2_weight, fc2_bias, act_fc2, N, 120, 84);
    sigmoid_inplace(act_fc2, N * 84);
    linear(act_fc2, fc3_weight, fc3_bias, act_fc3, N, 84, 10);
    cross_entropy_loss(act_fc3, train_labels + i, act_celoss, N, 10);

    // backward pass
    cross_entropy_loss_backward(act_celoss, train_labels + i, grad_celoss, N, 10);
    linear_backward_input(fc3_weight, grad_celoss, grad_fc3_input, N, 84, 10);
    linear_backward_weight(act_fc2, grad_celoss, grad_fc3_weight, N, 84, 10);
    linear_backward_bias(grad_celoss, grad_fc3_bias, N, 10);
    sigmoid_backward_inplace(act_fc2, grad_fc3_input, N * 84);
    linear_backward_input(fc2_weight, grad_fc3_input, grad_fc2_input, N, 120, 84);
    linear_backward_weight(act_fc1, grad_fc3_input, grad_fc2_weight, N, 120, 84);
    linear_backward_bias(grad_fc3_input, grad_fc2_bias, N, 84);
    sigmoid_backward_inplace(act_fc1, grad_fc2_input, N * 120);
    linear_backward_input(fc1_weight, grad_fc2_input, grad_fc1_input, N, 400, 120);
    linear_backward_weight(act_pool2, grad_fc2_input, grad_fc1_weight, N, 400, 120);
    linear_backward_bias(grad_fc2_input, grad_fc1_bias, N, 120);
    maxpool_backward(grad_fc1_input, act_conv2, grad_pool2, N, 16, 10, 10, 2, 2, 2);
    sigmoid_backward_inplace(act_conv2, grad_pool2, N * 16 * 10 * 10);
    convolution_backward_input(grad_pool2, conv2_weight, grad_conv2_input, N, 6, 14, 14, 16, 5, 5, 0);
    convolution_backward_weight(grad_pool2, act_pool1, grad_conv2_weight, N, 6, 14, 14, 16, 5, 5, 0);
    convolution_backward_bias(grad_pool2, grad_conv2_bias, N, 6, 14, 14, 16, 5, 5, 0);
    maxpool_backward(grad_conv2_input, act_conv1, grad_pool1, N, 6, 28, 28, 2, 2, 2);
    sigmoid_backward_inplace(act_conv1, grad_pool1, N * 6 * 28 * 28);
    // Note that we don't need to compute convolution_backward_input for conv1 because it is the first layer!
    convolution_backward_weight(grad_pool1, act_input, grad_conv1_weight, N, 1, 28, 28, 6, 5, 5, 2);
    convolution_backward_bias(grad_pool1, grad_conv1_bias, N, 1, 28, 28, 6, 5, 5, 2);

    // parameter update
    sgd(fc3_weight, grad_fc3_weight, learning_rate, 10 * 84);
    sgd(fc3_bias, grad_fc3_bias, learning_rate, 10);
    sgd(fc2_weight, grad_fc2_weight, learning_rate, 84 * 120);
    sgd(fc2_bias, grad_fc2_bias, learning_rate, 84);
    sgd(fc1_weight, grad_fc1_weight, learning_rate, 120 * 400);
    sgd(fc1_bias, grad_fc1_bias, learning_rate, 120);
    sgd(conv2_weight, grad_conv2_weight, learning_rate, 16 * 6 * 5 * 5);
    sgd(conv2_bias, grad_conv2_bias, learning_rate, 16);
    sgd(conv1_weight, grad_conv1_weight, learning_rate, 6 * 1 * 5 * 5);
    sgd(conv1_bias, grad_conv1_bias, learning_rate, 6);
  }
  printf("Training done. (elapsed = %f sec)\n", get_time() - start_time);
}

void inference(int num_inference_items, int num_inference_batch) {
  int N = num_inference_batch;

  printf("Inference...\n");
  int correct = 0;
  int progress_10th = 0;
  double start_time = get_time();
  for (int i = 0; i < num_inference_items; i += N) {
    if (i + N > num_inference_items) {
      N = num_inference_items - i;
    }
    if (i >= num_inference_items / 10 * (progress_10th + 1)) {
      ++progress_10th;
      printf("Inference progress: %d / %d\n", i , num_inference_items);
    }

    // forward pass
    uint8_to_float(test_images + i * 28 * 28, act_input, N * 28 * 28);
    convolution(act_input, conv1_weight, conv1_bias, act_conv1, N, 1, 28, 28, 6, 5, 5, 2);
    sigmoid_inplace(act_conv1, N * 6 * 28 * 28);
    maxpool(act_conv1, act_pool1, N, 6, 28, 28, 2, 2, 2);
    convolution(act_pool1, conv2_weight, conv2_bias, act_conv2, N, 6, 14, 14, 16, 5, 5, 0);
    sigmoid_inplace(act_conv2, N * 16 * 10 * 10);
    maxpool(act_conv2, act_pool2, N, 16, 10, 10, 2, 2, 2);
    linear(act_pool2, fc1_weight, fc1_bias, act_fc1, N, 16 * 5 * 5, 120);
    sigmoid_inplace(act_fc1, N * 120);
    linear(act_fc1, fc2_weight, fc2_bias, act_fc2, N, 120, 84);
    sigmoid_inplace(act_fc2, N * 84);
    linear(act_fc2, fc3_weight, fc3_bias, act_fc3, N, 84, 10);

    correct += check_answer(act_fc3, test_labels + i, N, 10);
  }
  printf("Inference done. (elapsed = %f sec) (%d correct out of %d)\n", get_time() - start_time, correct, num_inference_items);

}

void uint8_to_float(uint8_t* in, float* out, int N) {
  for (int i = 0; i < N; ++i) {
    out[i] = (float)in[i];
  }
}

int check_answer(float* pred, uint8_t* label, int N, int C) {
  // pred: (N, C)
  // label: (N)
  int correct = 0;
  for (int n = 0; n < N; ++n) {
    float max = -INFINITY;
    int max_idx = -1;
    for (int c = 0; c < C; ++c) {
      if (pred[n * C + c] > max) {
        max = pred[n * C + c];
        max_idx = c;
      }
    }
    if (max_idx == label[n]) {
      ++correct;
    }
  }
  return correct;
}

void convolution(float* in, float* weight, float* bias, float* out, int N, int C, int H, int W, int K, int R, int S, int pad) {
  // input : (N, C, H, W)
  // weight: (K, C, R, S)
  // bias  : (K)
  // output: (N, K, OH, OW)
  int OH = H - R + 2 * pad + 1;
  int OW = W - S + 2 * pad + 1;
  /*
   * IMPLEMENT HERE
   */
}

void sigmoid_inplace(float* inout, int N) {
  /*
   * IMPLEMENT HERE
   */
}

void maxpool(float* in, float* out, int N, int C, int H, int W, int R, int S, int stride) {
  // input : (N, C, H, W)
  // output: (N, C, OH, OW)
  int OH = (H - R) / stride + 1;
  int OW = (W - S) / stride + 1;
  /*
   * IMPLEMENT HERE
   */
}

void linear(float* in, float* weight, float* bias, float* output, int N, int C, int K) {
  // input : (N, C)
  // weight: (K, C)
  // bias  : (K)
  // output: (N, K)
  /*
   * IMPLEMENT HERE
   */
}

float cross_entropy_loss(float* pred, uint8_t* label, float* softmax_buf, int N, int C) {
  // pred: (N, C)
  // label: (N)
  // softmax_buf: (N, C)
  /*
   * IMPLEMENT HERE
   */
  float loss = 0.0f;
  return loss;
}

void convolution_backward_input(float* out_grad, float* weight, float* in_grad, int N, int C, int H, int W, int K, int R, int S, int pad) {
  // out_grad: (N, K, OH, OW)
  // weight: (K, C, R, S)
  // in_grad: (N, C, H, W)
  int OH = H - R + 2 * pad + 1;
  int OW = W - S + 2 * pad + 1;
  /*
   * IMPLEMENT HERE
   */
}

void convolution_backward_weight(float* out_grad, float* in, float* weight_grad, int N, int C, int H, int W, int K, int R, int S, int pad) {
  // out_grad: (N, K, OH, OW)
  // in: (N, C, H, W)
  // weight_grad: (K, C, R, S)
  int OH = H - R + 2 * pad + 1;
  int OW = W - S + 2 * pad + 1;
  /*
   * IMPLEMENT HERE
   */
}

void convolution_backward_bias(float* out_grad, float* bias_grad, int N, int C, int H, int W, int K, int R, int S, int pad) {
  // out_grad: (N, K, OH, OW)
  // bias_grad: (K)
  int OH = H - R + 2 * pad + 1;
  int OW = W - S + 2 * pad + 1;
  /*
   * IMPLEMENT HERE
   */
}

void sigmoid_backward_inplace(float* inout, float* inout_grad, int N) {
  /*
   * IMPLEMENT HERE
   */
}

void maxpool_backward(float* out_grad, float* in, float* in_grad, int N, int C, int H, int W, int R, int S, int stride) {
  // out_grad: (N, C, OH, OW)
  // in: (N, C, H, W)
  // in_grad: (N, C, H, W)
  int OH = (H - R) / stride + 1;
  int OW = (W - S) / stride + 1;
  /*
   * IMPLEMENT HERE
   */
}

void linear_backward_input(float* weight, float* out_grad, float* in_grad, int N, int C, int K) {
  // weight: (K, C)
  // out_grad: (N, K)
  // in_grad: (N, C)
  /*
   * IMPLEMENT HERE
   */
}

void linear_backward_weight(float* in, float* out_grad, float* weight_grad, int N, int C, int K) {
  // in: (N, C)
  // out_grad: (N, K)
  // weight_grad: (K, C)
  /*
   * IMPLEMENT HERE
   */
}

void linear_backward_bias(float* out_grad, float* bias_grad, int N, int K) {
  // out_grad: (N, K)
  // bias_grad: (K)
  /*
   * IMPLEMENT HERE
   */
}

void cross_entropy_loss_backward(float* softmax_buf, uint8_t* label, float* in_grad, int N, int C) {
  // softmax_buf: (N, C)
  // label: (N)
  // in_grad: (N, C)
  /*
   * IMPLEMENT HERE
   */
}

void sgd(float* weight, float* grad, float lr, int N) {
  for (int i = 0; i < N; ++i) {
    weight[i] -= lr * grad[i];
  }
}