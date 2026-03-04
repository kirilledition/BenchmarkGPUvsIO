cmake_minimum_required(VERSION 3.18)
project(BenchmarkGPUvsIO LANGUAGES CXX CUDA)

# ─── C++ / CUDA standards ──────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── generate_data  (pure C++) ─────────────────────────────────────────────
add_executable(generate_data generate_data.cpp)
target_compile_features(generate_data PRIVATE cxx_std_17)

# ─── run_benchmark  (CUDA) ─────────────────────────────────────────────────
add_executable(run_benchmark benchmark.cu)

# Target Ada Lovelace (RTX 40-series) for maximum instruction throughput.
# The -arch=sm_89 flag instructs nvcc to emit native SASS for that
# micro-architecture rather than generic PTX, which would require a
# costly JIT compilation step at runtime and distort benchmark timing.
# To target a different GPU, change the number (e.g. "86" for Ampere,
# "75" for Turing, "80" for A100).
set_target_properties(run_benchmark PROPERTIES
    CUDA_ARCHITECTURES "89"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(run_benchmark CUDA::cudart)
