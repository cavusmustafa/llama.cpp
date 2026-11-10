// ovcompile <xml> [device]: pure-OpenVINO compile_model reproducer; prints OVPHASE markers.
// Registers ov::op::internal::GatedDeltaNet so ggml-openvino IR dumps can be read.
#include <openvino/openvino.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/op/gated_delta_net.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
static void kmem(const char* n) {
    // xe DRM per-client memory + smaps rollup + system meminfo
    if (!getenv("KMEM")) return;
    DIR* d = opendir("/proc/self/fd"); struct dirent* e;
    while (d && (e = readdir(d))) {
        char lnk[256] = {0}; std::string p = std::string("/proc/self/fd/") + e->d_name;
        if (readlink(p.c_str(), lnk, sizeof(lnk) - 1) <= 0 || !strstr(lnk, "/dev/dri")) continue;
        std::ifstream f(std::string("/proc/self/fdinfo/") + e->d_name); std::string l;
        while (std::getline(f, l)) if (l.rfind("drm-total", 0) == 0 || l.rfind("drm-resident", 0) == 0 || l.rfind("drm-shared", 0) == 0)
            fprintf(stderr, "KMEM %s fd%s %s\n", n, e->d_name, l.c_str());
    }
    if (d) closedir(d);
    for (const char* fn : {"/proc/self/smaps_rollup", "/proc/meminfo"}) {
        std::ifstream f(fn); std::string l;
        while (std::getline(f, l)) for (const char* k : {"Rss:", "Anonymous:", "Shmem", "Shared_", "Private_", "MemAvailable", "Unevict", "Mlocked", "KReclaimable", "SUnreclaim", "Cached:", "Mapped:", "PageTables", "VmallocUsed"})
            if (l.rfind(k, 0) == 0) fprintf(stderr, "KMEM %s %s\n", n, l.c_str());
    }
}
static void mark(const char* n) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    fprintf(stderr, "OVPHASE %lld %s\n", (long long)ms, n); kmem(n);
}
int main(int argc, char** argv) {
    std::string dev = argc > 2 ? argv[2] : "GPU";
    ov::Core core;
    core.add_extension(std::make_shared<ov::OpExtension<ov::op::internal::GatedDeltaNet>>());
    mark("read_begin"); auto m = core.read_model(argv[1]); mark("read_end");
    if (getenv("CUT_EMBD_GATHER")) {
        // experiment: replace the token-embedding Gather by a Parameter, to see what the embedding costs
        for (auto& n : m->get_ordered_ops()) {
            auto g = std::dynamic_pointer_cast<ov::op::util::GatherBase>(n);
            if (!g || g->get_input_node_shared_ptr(0)->get_friendly_name() != "token_embd.weight") continue;
            auto par = std::make_shared<ov::op::v0::Parameter>(g->get_output_element_type(0), g->get_output_partial_shape(0));
            par->set_friendly_name("cut_embd");
            g->output(0).replace(par->output(0));
            m->add_parameters({par});
            fprintf(stderr, "CUT %s -> Parameter %s\n", g->get_friendly_name().c_str(), par->get_output_partial_shape(0).to_string().c_str());
        }
        m->validate_nodes_and_infer_types();
    }
    mark("compile_begin"); ov::AnyMap cfg;
    if (getenv("COMPILE_THREADS")) cfg[ov::compilation_num_threads.name()] = atoi(getenv("COMPILE_THREADS"));
    auto c = core.compile_model(m, dev, cfg); mark("compile_end");
    if (getenv("EXEC_GRAPH")) ov::serialize(c.get_runtime_model(), getenv("EXEC_GRAPH"));
    try {
        auto st = core.get_property(dev, "GPU_MEMORY_STATISTICS").as<std::map<std::string, uint64_t>>();
        for (auto& [k, v] : st) fprintf(stderr, "MEMSTAT %s %.2f GiB\n", k.c_str(), v / double(1 << 30));
    } catch (const std::exception& e) { fprintf(stderr, "memstat err %s\n", e.what()); }
    m.reset(); std::this_thread::sleep_for(std::chrono::seconds(3)); mark("after_del_model");
    c = {}; std::this_thread::sleep_for(std::chrono::seconds(3)); mark("after_del_compiled");
}
