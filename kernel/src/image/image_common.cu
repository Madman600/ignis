// ignis kernel leaf: the image ops' error channel (image_common.cuh) -- OURS (spec image-model/02,
// GitHub #319).

#include "image_common.cuh"

namespace ignis_image {

namespace {
thread_local std::string g_last_error = "no error";
}  // namespace

const char *last_error_cstr() { return g_last_error.c_str(); }

int32_t fail(const std::string &message) {
  g_last_error = message;
  return -1;
}

int32_t check_launch(const char *op) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return fail(std::string(op) + ": launch failed: " + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_image
