find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenCL REQUIRED)

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime openvino::threading OpenCL::OpenCL)

# Fused-MoE (GatherMatmulCompressed) prototype, OFF by default. The internal
# ov::op::internal::GatherMatmul lives in a header that the OpenVINO package does NOT
# export (ov_ops/gather_matmul.hpp, source tree only). Enable the experiment by pointing
# GGML_OPENVINO_OV_SRC at an OpenVINO source checkout; this adds the include dir and
# defines GGML_OPENVINO_HAVE_GATHER_MATMUL so the gated code compiles. Without it the
# backend builds normally and the fused path is compiled out entirely.
if (DEFINED GGML_OPENVINO_OV_SRC)
    target_include_directories(ggml-openvino PRIVATE ${GGML_OPENVINO_OV_SRC}/src/common/transformations/include)
    target_compile_definitions(ggml-openvino PRIVATE GGML_OPENVINO_HAVE_GATHER_MATMUL=1)
endif()

if (GGML_OPENVINO)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    else()
        message(FATAL_ERROR "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()
