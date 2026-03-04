# BenchmarkGPUvsIO

A GPU vs. I/O benchmarking pipeline that measures the real throughput of
each stage in a GWAS (Genome-Wide Association Study) data-processing loop:

| Phase | What is measured |
|-------|-----------------|
| A | Raw NVMe disk read (`O_DIRECT`, no page-cache) |
| B | Host → Device PCIe transfer (pinned memory + `cudaMemcpyAsync`) |
| C | GWAS dummy kernel: 2-bit unpack → matrix-vector multiply → threshold |
| D | End-to-end wall-clock time of the asynchronous double-buffered pipeline |

The pipeline uses **two CUDA streams** to overlap disk I/O, PCIe transfers,
and GPU compute.  If `T_disk > T_PCIe` and `T_disk > T_GPU` the system is
I/O bound and GPUDirect Storage should be investigated.

---

## Repository layout

```
BenchmarkGPUvsIO/
├── benchmark.cu        ← Main CUDA benchmarking application
├── generate_data.cpp   ← C++17 test-data generator (binary 2-bit genotypes)
├── CMakeLists.txt      ← CMake build (targets sm_89 / Ada Lovelace)
├── Dockerfile          ← Self-contained build environment
└── README.md
```

---

## Requirements

* **Docker** — only external dependency for the containerised path
* **NVIDIA driver** ≥ 525 (host)
* Enough free disk space on an NVMe volume for the test file (≥ 100 GiB)

For a native (non-Docker) build you additionally need:
* CUDA Toolkit 12.x (including `nvcc`)
* CMake ≥ 3.18
* GCC / G++ supporting C++17

---

## Quickstart (Docker — recommended)

### 1. Build the image

```bash
docker build -t benchmark_image .
```

### 2. Generate test data

Write the 100 GiB dummy file to a directory on your **physical NVMe drive**
(e.g. `/mnt/nvme/bench`).  Docker's internal overlay filesystem does not
support `O_DIRECT`, so the file **must** live on a bind-mounted host path.

```bash
docker run --rm \
  -v /mnt/nvme/bench:/data \
  benchmark_image \
  /app/build/generate_data /data/dummy_genotypes.bin 100
```

The second numeric argument is the file size in GiB (default: 100).

### 3. Run the benchmark

```bash
docker run --rm \
  --gpus all \
  -v /mnt/nvme/bench:/data \
  --shm-size=16g \
  --cap-add=SYS_NICE \
  benchmark_image \
  /app/build/run_benchmark /data/dummy_genotypes.bin
```

Flag reference:

| Flag | Purpose |
|------|---------|
| `--gpus all` | Pass the PCIe GPU device through to the container |
| `-v /path/to/nvme:/data` | Bind-mount NVMe directory; bypasses overlayfs so `O_DIRECT` works |
| `--shm-size=16g` | Raise shared-memory limit (default 64 MB) for large pinned buffers |
| `--cap-add=SYS_NICE` | Allow thread-affinity locking to prevent OS scheduler migration |

An optional second argument overrides the chunk size (default: 256 MiB):

```bash
/app/build/run_benchmark /data/dummy_genotypes.bin 512
```

---

## Native build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Executables are placed in `build/`.

To target a different GPU architecture, edit `CMakeLists.txt`:

```cmake
set_target_properties(run_benchmark PROPERTIES
    CUDA_ARCHITECTURES "86"   # e.g. "86" for RTX 30-series (Ampere)
)
```

---

## Sample output

```
────────────────────────────────────────────────────
  File        : /data/dummy_genotypes.bin
  Size        : 100.00 GiB
  Chunk size  : 256 MiB
  Chunks      : 400
  SNPs/chunk  : 1048576
  Direct I/O  : yes (O_DIRECT)
────────────────────────────────────────────────────
[ 400/ 400]  disk   4.21 GiB/s  pcie  12.50 GiB/s  gpu   14.372 TFLOPS

╔══════════════════════════════════════════════════════════╗
║                   BENCHMARK RESULTS                     ║
╠══════════════════════════════════════════════════════════╣
║  Raw Disk Read  (Phase A):            4.210 GiB/s ║
║  Pinned H→D Transfer  (Phase B):     12.500 GiB/s ║
║  Kernel Execution  (Phase C):        14.372 TFLOPS ║
║  Total Pipeline  (wall clock):       23.750 s      ║
╠══════════════════════════════════════════════════════════╣
║  T_disk  =   23.75 s                                   ║
║  T_PCIe  =    8.00 s                                   ║
║  T_GPU   =    0.60 s                                   ║
║  Bottleneck: DISK I/O bound  → evaluate GPUDirect Storage║
╚══════════════════════════════════════════════════════════╝
```

When `T_disk` dominates, the NVMe drive is the bottleneck.
When `T_PCIe` dominates, PCIe bandwidth is the bottleneck.
When `T_GPU`  dominates, the kernel is the bottleneck.

---

## Architecture notes

* **Double buffering** — two host pinned buffers and two device buffers allow
  `stream[0]` (transfers) and `stream[1]` (kernels) to work concurrently.
* **`O_DIRECT`** — skips the Linux page cache so the measured bandwidth
  reflects true NVMe throughput, not DRAM serving cached pages.
* **Pinned memory** — `cudaHostAlloc(cudaHostAllocDefault)` creates
  page-locked memory; the CUDA DMA engine can access it directly without an
  intermediate bounce copy.
* **CUDA events** — phase B and C timings use `cudaEventElapsedTime` for
  GPU-side precision; phase A uses `std::chrono::steady_clock`.
