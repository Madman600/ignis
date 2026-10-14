// ignis kernel leaf: host helpers shared by the image model's ops (spec image-model/02, GitHub
// #319) -- OURS. Leaf-internal: the error channel behind ignis_image_*_last_error and the
// per-device preparation record.
#ifndef IGNIS_IMAGE_COMMON_CUH
#define IGNIS_IMAGE_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ignis_image {

// The thread-local message of the most recent failure. Never NULL.
const char *last_error_cstr();
// Records `message` and returns -1.
int32_t fail(const std::string &message);
// 0, or -1 with the launch error recorded, for the kernel(s) just enqueued.
int32_t check_launch(const char *op);
// `p` is aligned to `bytes`.
inline bool aligned(const void *p, uintptr_t bytes) { return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0; }

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_COMMON_CUH
