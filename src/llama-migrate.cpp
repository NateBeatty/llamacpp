// llama-migrate.cpp
// Hot-swap model weight tensors between GPU and CPU without reloading from disk.
//
// See llama-migrate.h for integration instructions.

#include "llama-migrate.h"
#include "llama-model-impl.h"    // full llama_model::impl definition (includes llama-model.h + buft_list_t)
#include "llama-impl.h"          // for LLAMA_LOG_*

#include "ggml.h"
#include "ggml-cpp.h"            // ggml_context_ptr, ggml_backend_buffer_ptr
#include "ggml-alloc.h"          // ggml_tallocr
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers (file-local)
// ────────────────────────────────────────────────────────────────────────────

// Mirror of the static select_buft in llama-model.cpp: walks a buft_list and
// returns the first backend buffer type that supports an ADD(F32, F32) operation.
// Falls back to CPU if nothing is found.
static ggml_backend_buffer_type_t migrate_pick_buft(
        const buft_list_t &   buft_list,
        const llama_hparams & hparams) {

    for (const auto & [dev, buft] : buft_list) {
        // Probe: try to allocate a zero-byte buffer and check op support.
        ggml_backend_buffer_ptr probe { ggml_backend_buft_alloc_buffer(buft, 0) };
        if (!probe) continue;

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx { ggml_init(ctx_params) };
        if (!ctx) continue;

        ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hparams.n_embd);
        ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hparams.n_embd);
        ggml_tensor * op = ggml_add(ctx.get(), a, b);

        // Attach the probe buffer to src tensors so the device can inspect them.
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (op->src[i]) op->src[i]->buffer = probe.get();
        }

        if (ggml_backend_dev_supports_op(dev, op)) {
            return buft;
        }
    }
    return ggml_backend_cpu_buffer_type();
}

// ────────────────────────────────────────────────────────────────────────────
// Friend implementation – has full access to llama_model internals
// ────────────────────────────────────────────────────────────────────────────

bool llama_model_migrate_impl(struct llama_model * model, int32_t new_n_gpu_layers,
                              const llama_model_tensor_buft_override * overrides) {
    const int    n_layer = (int)model->hparams.n_layer;
    const size_t nd      = model->n_devices();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LLAMA_LOG_ERROR("%s: no CPU backend found\n", __func__);
        return false;
    }

    // ── 1. Recompute layer-to-device assignments ──────────────────────────

    const int new_resolved   = (new_n_gpu_layers < 0) ? n_layer + 1 : new_n_gpu_layers;
    const int i_gpu_start    = std::max(n_layer + 1 - new_resolved, 0);
    const int act_gpu_layers = (nd == 0) ? 0 : std::min(new_resolved, n_layer + 1);

    // Cumulative free-memory splits across GPU devices (same logic as load_tensors).
    std::vector<float> splits(nd, 0.0f);
    if (nd > 0) {
        const float * ts = model->params.tensor_split;
        bool all_zero = (ts == nullptr) ||
            std::all_of(ts, ts + nd, [](float x) { return x == 0.0f; });

        if (all_zero) {
            for (size_t i = 0; i < nd; ++i) {
                size_t total = 0, free_mem = 0;
                ggml_backend_dev_memory(model->devices[i].dev, &free_mem, &total);
                if (free_mem == 0 && total == 0) {
                    ggml_backend_dev_memory(cpu_dev, &free_mem, &total);
                }
                splits[i] = (float)free_mem;
            }
        } else {
            std::copy(ts, ts + nd, splits.begin());
        }

        float sum = 0.0f;
        for (float s : splits) sum += s;
        if (sum > 0.0f) {
            float running = 0.0f;
            for (float & s : splits) { running += s / sum; s = running; }
        } else {
            // Equal split fallback
            for (size_t i = 0; i < nd; ++i) splits[i] = float(i + 1) / float(nd);
        }
    }

    using layer_dev = llama_model::impl::layer_dev;

    auto get_layer_dev = [&](int il) -> layer_dev {
        if (nd == 0 || il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            return {cpu_dev, &model->pimpl->cpu_buft_list};
        }
        float frac    = float(il - i_gpu_start) / float(act_gpu_layers);
        int   gpu_idx = int(std::upper_bound(splits.begin(), splits.end(), frac) - splits.begin());
        gpu_idx       = std::min(gpu_idx, int(nd) - 1);
        auto * dev    = model->devices[gpu_idx].dev;
        return {dev, &model->pimpl->gpu_buft_list.at(dev)};
    };

    model->pimpl->dev_input = {cpu_dev, &model->pimpl->cpu_buft_list};
    for (int il = 0; il < n_layer; ++il) {
        model->pimpl->dev_layer[il] = get_layer_dev(il);
    }
    model->pimpl->dev_output = get_layer_dev(n_layer);

    // ── 2. Helper: pick buft for a given list ─────────────────────────────

    const llama_hparams & hp = model->hparams;

    auto pick_buft = [&](const buft_list_t & bl) -> ggml_backend_buffer_type_t {
        return migrate_pick_buft(bl, hp);
    };

    // Determine target buffer type for a tensor from its GGUF weight name.
    // Overrides (e.g. from --n-cpu-moe) are checked first; if matched, their
    // buffer type takes priority over the n_gpu_layers layer assignment.
    // Naming fallback:
    //   "blk.X.*"  → layer X  (dev_layer[X])
    //   "output.*" → dev_output
    //   everything else → CPU
    auto target_buft_for = [&](const char * name) -> ggml_backend_buffer_type_t {
        // Check tensor_buft_overrides first (same logic as llama-model-loader).
        if (overrides) {
            const std::string sname(name);
            for (const auto * ov = overrides; ov->pattern != nullptr; ++ov) {
                if (std::regex_search(sname, std::regex(ov->pattern))) {
                    if (ov->buft == ggml_backend_cpu_buffer_type()) {
                        return pick_buft(model->pimpl->cpu_buft_list);
                    }
                    return ov->buft;
                }
            }
        }
        if (strncmp(name, "blk.", 4) == 0) {
            int il = atoi(name + 4);
            if (il >= 0 && il < n_layer) {
                return pick_buft(*model->pimpl->dev_layer[il].buft_list);
            }
        }
        if (strncmp(name, "output", 6) == 0) {
            return pick_buft(*model->pimpl->dev_output.buft_list);
        }
        return pick_buft(model->pimpl->cpu_buft_list);
    };

    // ── 3. Migrate tensors that are on the wrong device ───────────────────

    std::vector<ggml_backend_buffer_ptr>       new_bufs;
    std::unordered_set<ggml_tensor *>           migrated_ptrs;
    std::unordered_set<ggml_backend_buffer_t>   touched_old;

    for (auto & [name, tensor] : model->tensors_by_name) {
        if (!tensor || !tensor->buffer || tensor->view_src) continue;
        // Skip if we already processed this exact tensor pointer (weight-tied models).
        if (!migrated_ptrs.insert(tensor).second) continue;

        const ggml_backend_buffer_type_t tgt = target_buft_for(name.c_str());
        if (!tgt || ggml_backend_buffer_get_type(tensor->buffer) == tgt) continue;

        // Save the current allocation so we can copy from it after reassignment.
        struct ggml_tensor src_snapshot = *tensor;

        // Allocate a fresh single-tensor buffer on the target device.
        // We pad to alignment so ggml_backend_tensor_alloc's size check passes.
        const size_t alloc_sz  = ggml_backend_buft_get_alloc_size(tgt, tensor);
        const size_t alignment = ggml_backend_buft_get_alignment(tgt);
        const size_t buf_sz    = GGML_PAD(alloc_sz, alignment);

        ggml_backend_buffer_t new_buf = ggml_backend_buft_alloc_buffer(tgt, buf_sz);
        if (!new_buf) {
            LLAMA_LOG_ERROR("%s: out of memory allocating buffer for tensor '%s'\n",
                            __func__, name.c_str());
            return false;
        }

        // ggml_backend_tensor_alloc asserts buffer==NULL and data==NULL.
        tensor->buffer = nullptr;
        tensor->data   = nullptr;

        // Assign tensor to the start of the new buffer.
        void * base = ggml_backend_buffer_get_base(new_buf);
        if (ggml_backend_tensor_alloc(new_buf, tensor, base) != GGML_STATUS_SUCCESS) {
            // Restore and bail; model is in a mixed state – caller may retry to CPU.
            tensor->buffer = src_snapshot.buffer;
            tensor->data   = src_snapshot.data;
            ggml_backend_buffer_free(new_buf);
            LLAMA_LOG_ERROR("%s: backend init failed for tensor '%s'\n",
                            __func__, name.c_str());
            return false;
        }

        ggml_backend_buffer_set_usage(new_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Transfer the weight data from the old backend to the new one.
        ggml_backend_tensor_copy(&src_snapshot, tensor);

        touched_old.insert(src_snapshot.buffer);
        new_bufs.emplace_back(new_buf);
    }

    // ── 4. Release old buffers that have no tensors left in them ──────────

    for (auto & [ctx_ptr, bufs] : model->pimpl->ctxs_bufs) {
        ggml_context * ctx = ctx_ptr.get();

        bufs.erase(
            std::remove_if(bufs.begin(), bufs.end(),
                [&](const ggml_backend_buffer_ptr & bp) -> bool {
                    if (!touched_old.count(bp.get())) return false;
                    // Keep if any tensor in this context still lives in this buffer.
                    for (auto * t = ggml_get_first_tensor(ctx); t;
                         t       = ggml_get_next_tensor(ctx, t)) {
                        if (t->buffer == bp.get()) return false;
                    }
                    return true;  // safe to free
                }),
            bufs.end()
        );
    }

    // ── 5. Hand new per-tensor buffers to the model for lifetime management

    if (!new_bufs.empty() && !model->pimpl->ctxs_bufs.empty()) {
        for (auto & b : new_bufs) {
            model->pimpl->ctxs_bufs[0].second.push_back(std::move(b));
        }
    }

    // ── 6. Persist the new GPU layer count in model params ────────────────

    model->params.n_gpu_layers = new_n_gpu_layers;

    LLAMA_LOG_INFO("%s: model weights migrated (n_gpu_layers = %d)\n",
                   __func__, new_n_gpu_layers);
    return true;
}

// ────────────────────────────────────────────────────────────────────────────
// Public C API
// ────────────────────────────────────────────────────────────────────────────

bool llama_model_migrate(struct llama_model * model, int32_t n_gpu_layers,
                         const llama_model_tensor_buft_override * tensor_buft_overrides) {
    return llama_model_migrate_impl(model, n_gpu_layers, tensor_buft_overrides);
}
