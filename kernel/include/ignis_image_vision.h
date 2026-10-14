/* ignis kernel leaf: the image model's vision tower -- Qwen3-VL-8B's (spec image-model/03, GitHub
 * #320). It is the 27B's tower program (kernel/src/vision_encode.cu, GitHub #178: the program
 * layer ours, every op the ADR 0010 vendored one) run with three parameters of its own instead of
 * a fork:
 *   - the merger writes 4096 wide (the 27B's 5120), into one contiguous buffer instead of the
 *     27B's embedding pool pages;
 *   - one call takes up to IGNIS_IMAGE_VISION_MAX_SEGMENTS attention segments (the condition
 *     images, one frame each), against the 27B's one item;
 *   - DeepStack: the residual after tower blocks deepstack_layers[i] (transformers'
 *     deepstack_visual_indexes, 8, 16 and 24) goes through DeepStack merger i, whose LayerNorm runs
 *     over the merged 4608 after the 2x2 view (use_postshuffle_norm), where the merger's runs over
 *     the hidden 1152 before it.
 * The text model adds the features through ignis_image_encoder.h's hooks: the merger's rows over
 * the image-pad rows, DeepStack merger i's after text layer i.
 *
 * Arithmetic is the 27B tower's: BF16 activations and residual, W8G32 weights through the
 * vendored ops::linear (A16), LayerNorm eps 1e-6, GELU-tanh in the blocks and GELU-exact in the
 * mergers, the 2-D vision RoPE at theta 10,000, attention non-causal within each segment.
 *
 * Every device call enqueues on `stream` (NULL: the legacy stream), allocates nothing on the device
 * and copies its host inputs into the workspace. Return 0, or -1 with the reason in
 * ignis_image_vision_last_error().
 */
#ifndef IGNIS_IMAGE_VISION_H
#define IGNIS_IMAGE_VISION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The geometry the leaf runs: Qwen3-VL-8B's vision tower (the 27B's but for the merger's width). */
#define IGNIS_IMAGE_VISION_BLOCKS 27u
#define IGNIS_IMAGE_VISION_HIDDEN 1152u
#define IGNIS_IMAGE_VISION_INTERMEDIATE 4304u
#define IGNIS_IMAGE_VISION_PATCH_DIM 1536u
#define IGNIS_IMAGE_VISION_MERGE_UNIT 4u
#define IGNIS_IMAGE_VISION_OUT_HIDDEN 4096u
#define IGNIS_IMAGE_VISION_DEEPSTACK 3u
/* The segment bound: ten condition images of one frame each (spec image-model/03). */
#define IGNIS_IMAGE_VISION_MAX_SEGMENTS 10u

/* One W8G32 matrix as the artifact stores it (row-split-k128-v1): codes int8 [rows][K padded to
 * 128] and scales FP16 [rows][padded K / 32], device pointers. */
struct ignis_image_vision_w8 {
  const void *codes;
  const void *scales;
};

/* One tower block: qkv [3456 x 1152], output [1152 x 1152], fc1 [4304 x 1152], fc2 [1152 x 4304]
 * (stored K 4352); the biases and both LayerNorms' weight and bias BF16. */
struct ignis_image_vision_block {
  struct ignis_image_vision_w8 qkv;
  const void *qkv_bias;
  struct ignis_image_vision_w8 output;
  const void *output_bias;
  struct ignis_image_vision_w8 fc1;
  const void *fc1_bias;
  struct ignis_image_vision_w8 fc2;
  const void *fc2_bias;
  const void *norm1_weight;
  const void *norm1_bias;
  const void *norm2_weight;
  const void *norm2_bias;
};

/* A merger: fc1 [4608 x 4608], fc2 [4096 x 4608], biases BF16; its LayerNorm BF16 [1152] (the
 * merger) or [4608] (a DeepStack merger). */
struct ignis_image_vision_merger {
  struct ignis_image_vision_w8 fc1;
  const void *fc1_bias;
  struct ignis_image_vision_w8 fc2;
  const void *fc2_bias;
  const void *norm_weight;
  const void *norm_bias;
};

/* The tower: the patch embedding [1152 x 1536] and its bias, the position table BF16 [2304][1152],
 * `block_count` (IGNIS_IMAGE_VISION_BLOCKS) blocks in a host array, the merger, and the DeepStack
 * mergers with the block after which each taps the residual (0-based, ascending, below
 * block_count). */
struct ignis_image_vision_weights {
  struct ignis_image_vision_w8 patch_embedding;
  const void *patch_embedding_bias;
  const void *position_embedding;
  const struct ignis_image_vision_block *blocks;
  uint32_t block_count;
  struct ignis_image_vision_merger merger;
  struct ignis_image_vision_merger deepstack[IGNIS_IMAGE_VISION_DEEPSTACK];
  uint32_t deepstack_layers[IGNIS_IMAGE_VISION_DEEPSTACK];
};

/* One call's patches and their control, host arrays, each image's as the 27B's media encode takes
 * it (the reference's VisionItemControl), the images one after the other:
 *   - patches BF16 [patches][1536], in 2x2 merge-block order (the processor's pixel_values);
 *   - position_ids int32 [2][patches]: every patch's row, then every patch's column;
 *   - cu_seqlens int32 [segments + 1]: 0, then each segment's end, ascending, ending at `patches`,
 *     every segment a positive multiple of IGNIS_IMAGE_VISION_MERGE_UNIT;
 *   - position_table_indices int32 and position_table_weights f32 [patches][4]: the bilinear
 *     corners in the 48 x 48 table and their weights.
 * `patches` is IGNIS_IMAGE_VISION_MERGE_UNIT x the merged tokens the call emits, at most 131,072
 * (32,768 tokens: the mergers' fc2 has no W8 arm above that). No segment has a bound of its own: a
 * condition image near 1024^2 at an extreme aspect ratio can exceed 1024 merged tokens, up to
 * 1,230 (96 x 13,120, after the pipeline's rounding and the processor's 200:1 limit); the caller's
 * workspace bounds the call. */
struct ignis_image_vision_input {
  uint32_t patches;
  uint32_t segments;
  const uint16_t *patch_rows;
  const int32_t *position_ids;
  const int32_t *cu_seqlens;
  const int32_t *position_table_indices;
  const float *position_table_weights;
};

/* The workspace a call of `tokens` merged tokens over `segments` segments needs, in bytes (the
 * 27B's layout: ignis_vision_workspace_bytes). 0 for tokens or segments 0, and for a size the
 * forward refuses: more than 32,768 tokens or IGNIS_IMAGE_VISION_MAX_SEGMENTS segments. */
uint64_t ignis_image_vision_workspace_bytes(uint32_t tokens, uint32_t segments);

/* Runs the tower on `input` (1 .. IGNIS_IMAGE_VISION_MAX_SEGMENTS segments): the merger's output
 * into `embeds`, BF16 [tokens][4096], and DeepStack merger i's into `deepstack[i]`, BF16
 * [tokens][4096] (NULL: that merger does not run). Merged token j is the j-th 2x2 block of the
 * patch rows, so the rows follow the images' image-pad tokens in sequence order. `workspace` holds
 * workspace_bytes(tokens, segments) or more, 256-byte aligned; the outputs are 16-byte aligned and
 * overlap neither each other nor the workspace. `deepstack` NULL: none of them runs. The host arrays
 * are read during the call (copied into the workspace); the device work is asynchronous on
 * `stream`. */
int32_t ignis_image_vision_forward(const struct ignis_image_vision_weights *weights,
                                   const struct ignis_image_vision_input *input, void *embeds,
                                   void *const *deepstack, void *workspace, uint64_t workspace_bytes,
                                   void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_vision_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_VISION_H */
