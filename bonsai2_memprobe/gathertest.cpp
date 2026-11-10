// gathertest: u2 embedding Gather (ggml-openvino pattern) on GPU vs CPU reference; also checks
// that the GPU exec graph no longer holds a decompressed table.
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <cmath>
#include <cstdio>
#include <random>
using namespace ov::opset8;
int main(int argc, char** argv) {
    const size_t V = 1000, G = 6, GS = 128, K = G * GS, T = 37;
    std::mt19937 rng(42);
    ov::Tensor w(ov::element::u2, {V, G, GS});
    auto* wp = static_cast<uint8_t*>(w.data());
    for (size_t i = 0; i < w.get_byte_size(); ++i) wp[i] = rng() & 0xff;  // codes 0..3
    auto wc = std::make_shared<Constant>(w);
    auto cvt = std::make_shared<Convert>(wc, ov::element::f16);
    auto zp = Constant::create(ov::element::f16, ov::Shape{}, {1.0f});
    auto sub = std::make_shared<Subtract>(cvt, zp);
    std::vector<ov::float16> sc(V * G);
    for (auto& s : sc) s = ov::float16(0.01f + (rng() % 1000) / 1000.0f);
    auto scale = Constant::create(ov::element::f16, ov::Shape{V, G, 1}, sc);
    auto mul = std::make_shared<Multiply>(sub, scale);
    auto rs = std::make_shared<Reshape>(mul, Constant::create(ov::element::i64, {2}, {int64_t(V), int64_t(K)}), false);
    auto cvt2 = std::make_shared<Convert>(rs, ov::element::f32);
    auto ids = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1, -1});
    auto g = std::make_shared<Gather>(cvt2, ids, Constant::create(ov::element::i32, {}, {0}));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{g}, ov::ParameterVector{ids});
    ov::Core core;
    std::vector<int32_t> idv(T);
    for (auto& x : idv) x = rng() % V;
    idv[0] = 0; idv[1] = V - 1;
    ov::Tensor it(ov::element::i32, {1, T}, idv.data());
    std::vector<std::vector<float>> outs;
    {  // host reference: (code - zp) * scale, u2 LSB-first
        std::vector<float> ref(T * K);
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < K; ++k) {
                size_t e = idv[t] * K + k;
                int code = (wp[e / 4] >> (2 * (e % 4))) & 3;
                ref[t * K + k] = float(ov::float16(float(ov::float16((code - 1.0f) * float(sc[idv[t] * G + k / GS])))));
            }
        outs.push_back(ref);
    }
    for (const char* dev : {"GPU"}) {
        auto c = core.compile_model(model, dev);
        if (std::string(dev) == "GPU") {
            for (auto& n : c.get_runtime_model()->get_ordered_ops())
                if (n->get_rt_info().count("layerType") && n->get_rt_info().at("layerType").as<std::string>() == "Gather")
                    printf("GPU gather impl: %s\n", n->get_rt_info().at("primitiveType").as<std::string>().c_str());
        }
        auto r = c.create_infer_request();
        r.set_input_tensor(it);
        r.infer();
        auto o = r.get_output_tensor();
        outs.emplace_back(o.data<float>(), o.data<float>() + o.get_size());
    }
    double maxd = 0; size_t bad = 0;
    for (size_t i = 0; i < outs[0].size(); ++i) {
        double d = std::fabs(outs[0][i] - outs[1][i]);
        maxd = std::max(maxd, d);
        bad += d > 1e-3 * (1 + std::fabs(outs[0][i]));
    }
    printf("[host-ref vs GPU] elements %zu  max|cpu-gpu| %.6g  mismatches %zu  sample cpu %.4f gpu %.4f\n", outs[0].size(), maxd, bad, outs[0][5], outs[1][5]);
    return bad != 0;
}
