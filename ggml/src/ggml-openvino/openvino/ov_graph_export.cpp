#include "ov_graph_export.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <openvino/core/node.hpp>

namespace ov {
namespace frontend {
namespace ggml {

namespace {

void json_escape(std::ostream & os, const std::string & s) {
    os << '"';
    for (char c : s) {
        switch (c) {
        case '"':  os << "\\\""; break;
        case '\\': os << "\\\\"; break;
        case '\n': os << "\\n";  break;
        case '\r': os << "\\r";  break;
        case '\t': os << "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                os << buf;
            } else {
                os << c;
            }
        }
    }
    os << '"';
}

template <typename T>
std::string to_str(const T & v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

}  // namespace

void dump_ov_model_json(const std::shared_ptr<const ov::Model> & model, const std::string & path_base,
                        const std::string & stage) {
    if (!model) {
        return;
    }

    const auto ops = model->get_ordered_ops();

    // assign a stable index to every node so edges can reference them
    std::unordered_map<const ov::Node *, size_t> id_of;
    id_of.reserve(ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
        id_of[ops[i].get()] = i;
    }

    const std::string filename = path_base + "." + stage + ".json";
    std::ofstream fp(filename);
    if (!fp) {
        return;
    }

    fp << "{\n";
    fp << "  \"stage\": ";  json_escape(fp, stage);  fp << ",\n";
    fp << "  \"model_name\": ";  json_escape(fp, model->get_friendly_name());  fp << ",\n";
    fp << "  \"n_nodes\": " << ops.size() << ",\n";
    fp << "  \"nodes\": [\n";

    for (size_t i = 0; i < ops.size(); ++i) {
        const auto & node = ops[i];
        fp << "    {\n";
        fp << "      \"id\": " << i << ",\n";
        fp << "      \"name\": ";  json_escape(fp, node->get_friendly_name());  fp << ",\n";
        fp << "      \"type\": ";  json_escape(fp, std::string(node->get_type_name()));  fp << ",\n";

        // runtime (executable) graphs fuse and rename nodes; these rt_info keys carry
        // the original layer name(s) and exec metadata used to correlate back to ggml ops.
        const auto & rt = node->get_rt_info();
        for (const char * key : {"originalLayersNames", "layerType", "execTimeMcs", "outputLayouts", "primitiveType"}) {
            auto it = rt.find(key);
            if (it != rt.end()) {
                fp << "      \"" << key << "\": ";
                json_escape(fp, it->second.as<std::string>());
                fp << ",\n";
            }
        }

        // outputs: shape + element type per port
        fp << "      \"outputs\": [";
        for (size_t p = 0; p < node->get_output_size(); ++p) {
            if (p) {
                fp << ", ";
            }
            fp << "{\"port\": " << p
               << ", \"shape\": ";  json_escape(fp, to_str(node->get_output_partial_shape(p)));
            fp << ", \"etype\": ";  json_escape(fp, to_str(node->get_output_element_type(p)));
            fp << "}";
        }
        fp << "],\n";

        // inputs: edge sources (source node id + source port)
        fp << "      \"inputs\": [";
        for (size_t in = 0; in < node->get_input_size(); ++in) {
            if (in) {
                fp << ", ";
            }
            const auto src_out = node->get_input_source_output(in);
            const auto * src_node = src_out.get_node();
            auto it = id_of.find(src_node);
            const long src_id = it == id_of.end() ? -1 : static_cast<long>(it->second);
            fp << "{\"port\": " << in
               << ", \"src_id\": " << src_id
               << ", \"src_port\": " << src_out.get_index() << "}";
        }
        fp << "]\n";

        fp << "    }" << (i + 1 < ops.size() ? "," : "") << "\n";
    }

    fp << "  ]\n";
    fp << "}\n";
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
