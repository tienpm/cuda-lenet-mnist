#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include "lenet.h"

// globals
static const char* all_in_one_bin_filename = nullptr;
static bool mode_training = false;
static int num_training_epochs = 1;
static int num_training_items = 60000;
static int num_training_batch = 100;
static int num_inference_items = 10000;
static int num_inference_batch = 100;
static float learning_rate = 0.1f;

static void print_help(const char* prog_name) {
  printf("Usage: %s [options] [all_in_one.bin]\n", prog_name);
  printf("Options:\n");
  printf("  -t : Training mode. (default: off)\n");
  printf("  -e : Training epoch. (default: 1)\n");
  printf("  -a : Training items. (default: 60,000)\n");
  printf("  -b : Training batch size. (default: 100)\n");
  printf("  -c : Inference items. (default: 10,000)\n");
  printf("  -d : Inference batch size. (default: 100)\n");
  printf("  -l : Learning rate. (default: 0.1)\n");
  printf("  -h : print this page.\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "te:a:b:c:d:l:h")) != -1) {
    switch (c) {
      case 't': mode_training = true; break;
      case 'e': num_training_epochs = atoi(optarg); break;
      case 'a': num_training_items = atoi(optarg); break;
      case 'b': num_training_batch = atoi(optarg); break;
      case 'c': num_inference_items = atoi(optarg); break;
      case 'd': num_inference_batch = atoi(optarg); break;
      case 'l': learning_rate = atof(optarg); break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: all_in_one_bin_filename = argv[i]; break;
      default: break;
    }
  }
  if (all_in_one_bin_filename == nullptr) {
    printf("all_in_one filename is not specified.\n");
    print_help(argv[0]);
    exit(1);
  }
  printf("Options:\n");
  printf("  all_in_one filename: %s\n", all_in_one_bin_filename);
  printf("  Mode: %s\n", mode_training ? "training" : "inference");
  printf("  Training epochs: %d\n", num_training_epochs);
  printf("  Training items: %d\n", num_training_items);
  printf("  Training batch size: %d\n", num_training_batch);
  printf("  Inference items: %d\n", num_inference_items);
  printf("  Inference batch size: %d\n", num_inference_batch);
  printf("  Learning rate: %f\n", learning_rate);
  printf("\n");
}

static void* read_all_data(const char* filename) {

  // Check file size is 55127704-byte
  printf("Opening %s...\n", filename);
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("Failed to open %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  if (size != ALL_IN_ONE_BIN_BYTES) {
    printf("File size is not %ld-byte\n", ALL_IN_ONE_BIN_BYTES);
    exit(1);
  }
  fseek(fp, 0, SEEK_SET);
  void* buffer = malloc(size);
  size_t res = fread(buffer, 1, size, fp);
  if (res != ALL_IN_ONE_BIN_BYTES) {
    printf("fread only read %ld-byte (out of %ld-byte)\n", res, ALL_IN_ONE_BIN_BYTES);
    exit(1);
  }

  printf("Model and dataset are loaded.\n");
  return buffer;
}

int main(int argc, char** argv) {
  parse_opt(argc, argv);

  void* all_in_one_buffer = read_all_data(all_in_one_bin_filename);

  initialize(all_in_one_buffer, mode_training, std::max(num_training_batch, num_inference_batch));

  if (mode_training) {
    for (int i = 0; i < num_training_epochs; ++i) {
      printf("Epoch %d / %d\n", i + 1, num_training_epochs);
      train(num_training_items, num_training_batch, learning_rate);
      inference(num_inference_items, num_inference_batch);
    }
  } else {
    inference(num_inference_items, num_inference_batch);
  }

  finalize(mode_training);

  free(all_in_one_buffer);

  return 0;
}