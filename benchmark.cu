// benchmark.cu — GPU GWAS I/O Benchmarking Pipeline
//
// Implements an asynchronous double-buffered pipeline across four phases:
//
//   Phase A: Raw disk read with O_DIRECT (bypasses OS page cache).
//   Phase B: Host→Device transfer using pinned (page-locked) memory and
//            cudaMemcpyAsync (stream 0).
//   Phase C: GWAS dummy kernel — unpack 2-bit PLINK genotypes, compute
//            y = X·β, apply threshold, write output (stream 1).
//   Phase D: Two cudaStream_t streams are used to overlap compute with
//            transfers; cudaStreamSynchronize guards ring-buffer reuse.
//
// Usage:
//   run_benchmark <binary_file> [chunk_mb]
//
// chunk_mb defaults to 256 (256 MiB per chunk).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// ─── Compile-time constants ───────────────────────────────────────────────────

// Number of biological samples packed into one SNP row.
// Changing this requires recompiling.
static constexpr int N_SAMPLES       = 1024;

// PLINK packs 4 samples per byte → bytes needed to store one SNP row.
static constexpr int BYTES_PER_SNP   = N_SAMPLES / 4;  // 256

// One CUDA thread per packed byte within a SNP block.
static constexpr int THREADS_PER_SNP = BYTES_PER_SNP;  // 256

// Association score threshold for the dummy statistical filter.
static constexpr float THRESHOLD     = 0.5f;

// Default I/O chunk size (can be overridden on the command line).
static constexpr size_t DEFAULT_CHUNK_MB = 256;

// ─── CUDA error helper ────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Device helper ───────────────────────────────────────────────────────────

// Map a 2-bit PLINK genotype code to a 32-bit float dosage value.
//
// PLINK BED 2-bit encoding (LSB-first, per sample):
//   0b00 (0) — homozygous first allele  → 0.0
//   0b01 (1) — missing genotype         → 0.0  (mean-imputed as 0)
//   0b10 (2) — heterozygous             → 1.0
//   0b11 (3) — homozygous second allele → 2.0
__device__ __forceinline__
float plink_to_float(unsigned bits) {
    return (bits < 2u) ? 0.0f : static_cast<float>(bits - 1u);
}

// ─── GWAS dummy kernel (Phase C) ─────────────────────────────────────────────
//
// Grid  : (n_snps, 1, 1)   — one block per SNP row
// Block : (256, 1, 1)      — one thread per packed byte  (= 4 samples)
//
// Per-thread steps:
//   1. Read one packed byte → unpack 4 × 2-bit PLINK genotypes → 32-bit floats.
//   2. Compute partial dot-product with the corresponding beta slice (y = X·β).
// Block-level steps:
//   3. Parallel-reduction tree to accumulate the full dot-product across all
//      256 threads (all 1024 samples).
//   4. Thread 0 applies a threshold and writes the association score.

__global__ void gwas_kernel(
        const uint8_t* __restrict__ packed,   // [n_snps × BYTES_PER_SNP]
        const float*   __restrict__ beta,     // [N_SAMPLES]
        float*         __restrict__ output,   // [n_snps]
        int n_snps)
{
    const int snp = blockIdx.x;   // SNP index
    const int tid = threadIdx.x;  // byte index within SNP (0…255)

    if (snp >= n_snps) return;

    __shared__ float sdata[THREADS_PER_SNP];

    // ── Step 1 & 2: unpack + partial dot-product ─────────────────────────
    const uint8_t byte = packed[(size_t)snp * BYTES_PER_SNP + tid];
    const int     base = tid * 4;  // first sample index for this thread

    const float g0 = plink_to_float( byte        & 0x3u);
    const float g1 = plink_to_float((byte >> 2u) & 0x3u);
    const float g2 = plink_to_float((byte >> 4u) & 0x3u);
    const float g3 = plink_to_float((byte >> 6u) & 0x3u);

    sdata[tid] = g0 * beta[base]     + g1 * beta[base + 1]
               + g2 * beta[base + 2] + g3 * beta[base + 3];

    __syncthreads();

    // ── Step 3: parallel tree reduction ──────────────────────────────────
    for (int s = THREADS_PER_SNP / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // ── Step 4: threshold and write ───────────────────────────────────────
    if (tid == 0) {
        output[snp] = (sdata[0] > THRESHOLD) ? sdata[0] : 0.0f;
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

static double wall_sec()
{
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <binary_file> [chunk_mb]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char*  path        = argv[1];
    const size_t chunk_mb    = (argc >= 3)
                               ? static_cast<size_t>(std::stoull(argv[2]))
                               : DEFAULT_CHUNK_MB;
    const size_t chunk_bytes = chunk_mb * (1024ULL * 1024);

    if (chunk_bytes % BYTES_PER_SNP != 0) {
        fprintf(stderr,
                "Error: chunk size (%zu bytes) must be a multiple of "
                "BYTES_PER_SNP (%d).\n",
                chunk_bytes, BYTES_PER_SNP);
        return EXIT_FAILURE;
    }

    const size_t n_snps = chunk_bytes / BYTES_PER_SNP;

    // ── Phase A: open file with O_DIRECT ─────────────────────────────────
    // O_DIRECT bypasses the OS page cache; the kernel transfers data directly
    // between the NVMe controller and the user-space buffer.  Requirements:
    //   • buffer address  aligned to logical block size (≥ 512 B, typically 4 KiB)
    //   • file offset     aligned to logical block size
    //   • transfer length aligned to logical block size
    // cudaHostAlloc returns page-aligned (4096 B) memory, and chunk_bytes is a
    // multiple of 4 MiB, so all three requirements are satisfied.
    bool using_direct_io = false;

#ifdef O_DIRECT
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        using_direct_io = true;
    } else {
        // Fall back for filesystems that do not support O_DIRECT (e.g.
        // overlayfs inside Docker without a bind-mounted NVMe volume).
        fprintf(stderr,
                "[warn] O_DIRECT unavailable (%s) — falling back to "
                "buffered I/O.\n       For accurate disk benchmarks, run "
                "with a bind-mounted NVMe volume (see README).\n",
                strerror(errno));
        fd = open(path, O_RDONLY);
    }
#else
    int fd = open(path, O_RDONLY);
#endif

    if (fd < 0) { perror("open"); return EXIT_FAILURE; }

    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return EXIT_FAILURE; }

    const size_t file_bytes = static_cast<size_t>(st.st_size);
    const size_t n_chunks   = file_bytes / chunk_bytes;  // ignore partial tail

    printf("────────────────────────────────────────────────────\n");
    printf("  File        : %s\n", path);
    printf("  Size        : %.2f GiB\n",  file_bytes  / (double)(1ULL << 30));
    printf("  Chunk size  : %zu MiB\n",   chunk_mb);
    printf("  Chunks      : %zu\n",       n_chunks);
    printf("  SNPs/chunk  : %zu\n",       n_snps);
    printf("  Direct I/O  : %s\n",        using_direct_io ? "yes (O_DIRECT)" : "no (buffered)");
    printf("────────────────────────────────────────────────────\n");

    if (n_chunks == 0) {
        fprintf(stderr, "File is smaller than one chunk (%zu MiB).\n", chunk_mb);
        return EXIT_FAILURE;
    }

    // ── Phase B: allocate pinned host ring buffers ────────────────────────
    // cudaHostAllocDefault creates page-locked (pinned) memory.  Pinned memory
    // allows the CUDA DMA engine to transfer without an intermediate copy,
    // avoiding the hidden synchronous memcpy that non-pinned memory requires.
    uint8_t* h_buf[2];
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaHostAlloc(&h_buf[i], chunk_bytes, cudaHostAllocDefault));

    // Device buffers (double-buffered to match host ring).
    uint8_t* d_packed[2];
    float*   d_output[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_packed[i], chunk_bytes));
        CUDA_CHECK(cudaMalloc(&d_output[i], n_snps * sizeof(float)));
    }

    // Beta vector: pre-generated effect-size estimates stored in VRAM.
    float* d_beta;
    CUDA_CHECK(cudaMalloc(&d_beta, N_SAMPLES * sizeof(float)));
    {
        std::vector<float> h_beta(N_SAMPLES);
        for (int i = 0; i < N_SAMPLES; ++i)
            h_beta[i] = 1e-3f * static_cast<float>(i % 100 + 1);
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(),
                              N_SAMPLES * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ── Phase D: create two CUDA streams ─────────────────────────────────
    // stream[0]: Host → Device PCIe transfers
    // stream[1]: GWAS compute kernels
    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    // CUDA events for accurate per-phase GPU-side timing.
    cudaEvent_t ev_xfer_start, ev_xfer_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;
    CUDA_CHECK(cudaEventCreate(&ev_xfer_start));
    CUDA_CHECK(cudaEventCreate(&ev_xfer_stop));
    CUDA_CHECK(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK(cudaEventCreate(&ev_kern_stop));

    // Kernel launch dimensions.
    const dim3 grid(static_cast<unsigned>(n_snps));
    const dim3 block(THREADS_PER_SNP);

    // Timing accumulators.
    double    t_disk   = 0.0;
    double    t_pcie   = 0.0;
    double    t_kernel = 0.0;
    long long total_flops   = 0;
    size_t    disk_bytes_read = 0;  // tracks actual bytes fetched from disk

    const double t_pipeline_start = wall_sec();

    // ── Pre-read chunk 0 (Phase A) ────────────────────────────────────────
    {
        const double t0 = wall_sec();
        if (pread(fd, h_buf[0], chunk_bytes, 0) != static_cast<ssize_t>(chunk_bytes)) {
            perror("pread chunk 0"); return EXIT_FAILURE;
        }
        t_disk += wall_sec() - t0;
        disk_bytes_read += chunk_bytes;
    }

    // ── Initial transfer: chunk 0 → d_packed[0] (Phase B) ────────────────
    {
        CUDA_CHECK(cudaEventRecord(ev_xfer_start, stream[0]));
        CUDA_CHECK(cudaMemcpyAsync(d_packed[0], h_buf[0], chunk_bytes,
                                   cudaMemcpyHostToDevice, stream[0]));
        CUDA_CHECK(cudaEventRecord(ev_xfer_stop, stream[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_xfer_start, ev_xfer_stop));
        t_pcie += static_cast<double>(ms) * 1e-3;
    }

    // ── Pipeline loop ─────────────────────────────────────────────────────
    //
    // At the start of iteration k, d_packed[k%2] is ready with chunk k.
    //
    // Within each iteration the following operations overlap:
    //
    //   stream[1] : kernel on d_packed[cur]           (Phase C)
    //   CPU       : pread chunk k+1 → h_buf[nxt]      (Phase A)
    //   stream[0] : cudaMemcpyAsync h_buf[nxt] →
    //               d_packed[nxt]                      (Phase B)
    //
    // cudaStreamSynchronize(stream[0]) prevents h_buf[nxt] from being
    // overwritten before the PCIe transfer has consumed it.
    // cudaStreamSynchronize(stream[1]) prevents d_packed[cur] from being
    // overwritten by a future transfer before the kernel finishes reading it.

    for (size_t k = 0; k < n_chunks; ++k) {
        const int cur = static_cast<int>(k & 1);
        const int nxt = cur ^ 1;

        // ── Phase C: launch kernel on current chunk ───────────────────
        CUDA_CHECK(cudaEventRecord(ev_kern_start, stream[1]));
        gwas_kernel<<<grid, block, 0, stream[1]>>>(
                d_packed[cur], d_beta, d_output[cur],
                static_cast<int>(n_snps));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_kern_stop, stream[1]));

        // ── Phase A + B: prepare next chunk while kernel runs ─────────
        if (k + 1 < n_chunks) {
            // Phase A: CPU disk read (overlaps with stream[1] kernel).
            const double t0 = wall_sec();
            if (pread(fd, h_buf[nxt], chunk_bytes,
                      static_cast<off_t>((k + 1) * chunk_bytes))
                != static_cast<ssize_t>(chunk_bytes)) {
                perror("pread next chunk"); return EXIT_FAILURE;
            }
            t_disk += wall_sec() - t0;
            disk_bytes_read += chunk_bytes;

            // Phase B: PCIe transfer (may overlap with tail of kernel).
            CUDA_CHECK(cudaEventRecord(ev_xfer_start, stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(d_packed[nxt], h_buf[nxt], chunk_bytes,
                                       cudaMemcpyHostToDevice, stream[0]));
            CUDA_CHECK(cudaEventRecord(ev_xfer_stop, stream[0]));
            // cudaStreamSynchronize guards h_buf[nxt] from premature reuse.
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_xfer_start, ev_xfer_stop));
            t_pcie += static_cast<double>(ms) * 1e-3;
        }

        // cudaStreamSynchronize guards d_packed[cur] from being overwritten
        // by a future transfer before this iteration's kernel is done.
        CUDA_CHECK(cudaStreamSynchronize(stream[1]));
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_kern_start, ev_kern_stop));
            t_kernel += static_cast<double>(ms) * 1e-3;
        }

        // Each SNP costs 2 × N_SAMPLES FLOPs (fused multiply-add ×
        // N_SAMPLES, plus the reduction tree which is O(N_SAMPLES)).
        total_flops += static_cast<long long>(n_snps) * N_SAMPLES * 2;

        if (k % 10 == 0 || k + 1 == n_chunks) {
            const size_t computed = (k + 1) * chunk_bytes;
            printf("\r[%4zu/%4zu]  disk %6.2f GiB/s  "
                   "pcie %6.2f GiB/s  "
                   "gpu %7.3f TFLOPS   ",
                   k + 1, n_chunks,
                   (disk_bytes_read / (double)(1ULL << 30)) / t_disk,
                   (computed        / (double)(1ULL << 30)) / t_pcie,
                   static_cast<double>(total_flops) / (t_kernel * 1e12));
            fflush(stdout);
        }
    }

    const double t_total     = wall_sec() - t_pipeline_start;
    const double bytes_total = static_cast<double>(n_chunks) *
                               static_cast<double>(chunk_bytes);

    // ── Final diagnostic report ───────────────────────────────────────────
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║                   BENCHMARK RESULTS                     ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  %-34s  %8.3f GiB/s  ║\n",
           "Raw Disk Read  (Phase A):",
           (bytes_total / (double)(1ULL << 30)) / t_disk);
    printf("║  %-34s  %8.3f GiB/s  ║\n",
           "Pinned H→D Transfer  (Phase B):",
           (bytes_total / (double)(1ULL << 30)) / t_pcie);
    printf("║  %-34s  %8.3f TFLOPS ║\n",
           "Kernel Execution  (Phase C):",
           static_cast<double>(total_flops) / (t_kernel * 1e12));
    printf("║  %-34s  %8.3f s      ║\n",
           "Total Pipeline  (wall clock):", t_total);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  T_disk  = %8.3f s                                   ║\n", t_disk);
    printf("║  T_PCIe  = %8.3f s                                   ║\n", t_pcie);
    printf("║  T_GPU   = %8.3f s                                   ║\n", t_kernel);
    {
        const double max_t = std::max({t_disk, t_pcie, t_kernel});
        const char* verdict =
            (max_t == t_disk)   ? "DISK I/O bound  → evaluate GPUDirect Storage" :
            (max_t == t_pcie)   ? "PCIe bandwidth bound"                         :
                                   "GPU compute bound";
        printf("║  Bottleneck: %-44s║\n", verdict);
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");

    // ── Cleanup ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(ev_xfer_start));
    CUDA_CHECK(cudaEventDestroy(ev_xfer_stop));
    CUDA_CHECK(cudaEventDestroy(ev_kern_start));
    CUDA_CHECK(cudaEventDestroy(ev_kern_stop));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
        CUDA_CHECK(cudaFreeHost(h_buf[i]));
        CUDA_CHECK(cudaFree(d_packed[i]));
        CUDA_CHECK(cudaFree(d_output[i]));
    }
    CUDA_CHECK(cudaFree(d_beta));
    close(fd);

    return EXIT_SUCCESS;
}
