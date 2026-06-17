#pragma once

#include <memory>
#include <string>

#include <openvino/core/model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

// Serialize an ov::Model's structure (nodes, ports, edges - no weight data) to a
// JSON file for the graph viewer. Used to inspect the backend-stage translation
// before the device plugin runs its own passes.
//
// stage labels the graph (e.g. "translated", "post_backend_passes", "runtime");
// it is written into the JSON and appended to the filename.
void dump_ov_model_json(const std::shared_ptr<const ov::Model> & model, const std::string & path_base,
                        const std::string & stage);

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
