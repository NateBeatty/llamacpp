#pragma once

// llama-migrate: hot-swap model weights between GPU and CPU without reloading from disk.
//
// How to integrate into another fork:
//   1. Add the following friend declaration inside the llama_model class (llama-model.h):
//        friend bool llama_model_migrate_impl(struct llama_model * model, int32_t new_n_gpu_layers);
//   2. Add llama-migrate.cpp to your build system.
//   3. Include this header where needed.
//
// Usage:
//   // Move all weights to CPU (frees VRAM):
//   llama_model_migrate(model, 0);
//   llama_free(ctx);  // context must be recreated after migration
//   ctx = llama_init_from_model(model, ctx_params);
//
//   // Move weights back to GPU (last 32 layers):
//   llama_model_migrate(model, 32);
//   llama_free(ctx);
//   ctx = llama_init_from_model(model, ctx_params);

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Migrate model weight tensors to a different GPU/CPU configuration.
//
// All llama_context objects using this model MUST be freed before calling this
// function and recreated afterwards (the scheduler needs to reflect the new layout).
//
// n_gpu_layers semantics match llama_model_params::n_gpu_layers:
//   0  = all weights on CPU
//   -1 = all weights on GPU
//   N  = last N transformer layers on GPU (same as loading with n_gpu_layers=N)
//
// tensor_buft_overrides: optional null-terminated array of tensor buffer type overrides
//   (same format as llama_model_params::tensor_buft_overrides). Overrides are applied
//   before the n_gpu_layers logic, so e.g. MoE expert tensors pinned to CPU via
//   --n-cpu-moe will remain on CPU even after a GPU migration. Pass NULL if none.
//
// Returns true on success, false if a backend allocation fails (e.g. insufficient VRAM).
// On failure the model is left in a consistent state with a partial migration; call
// llama_model_migrate(model, 0, NULL) to safely fall back to CPU-only.
LLAMA_API bool llama_model_migrate(struct llama_model * model, int32_t n_gpu_layers,
                                   const struct llama_model_tensor_buft_override * tensor_buft_overrides);

#ifdef __cplusplus
}
#endif
