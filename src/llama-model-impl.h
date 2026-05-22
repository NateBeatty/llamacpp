#pragma once
// Internal header: full definition of llama_model::impl.
// Only include within src/ — not part of the public API.

#include "llama-model.h"
#include "llama-model-loader.h" // buft_list_t (also pulls in llama-mmap.h, ggml-cpp.h)

#include <map>
#include <string>
#include <vector>

struct llama_model::impl {
    impl()  = default;
    ~impl() = default;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t  dev;
        buft_list_t       * buft_list;
    };

    layer_dev dev_input  = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};
