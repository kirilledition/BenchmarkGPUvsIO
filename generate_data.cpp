// generate_data.cpp
// Generates a flat binary file of random 2-bit-packed genotypes.
//
// Each byte holds four 2-bit samples (LSB-first), mirroring the PLINK BED
// format layout.  All 256 bit-patterns are valid 2-bit states, so the file
// is simply filled with pseudo-random bytes.
//
// Usage:
//   generate_data <output_path> [size_gb]
//
// Defaults: output_path = dummy_genotypes.bin, size_gb = 100
//
// The generated file size is always an exact multiple of 256 MiB (the
// default benchmark chunk size), so every chunk boundary is 4096-byte
// aligned — required for O_DIRECT.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static constexpr size_t WRITE_CHUNK = 64ULL << 20;  // 64 MiB per write call

int main(int argc, char* argv[])
{
    const char*  path    = (argc >= 2) ? argv[1] : "dummy_genotypes.bin";
    const size_t size_gb = (argc >= 3) ? std::stoull(argv[2]) : 100ULL;

    // File size in bytes.  1 GiB = 16 × 64 MiB, so total is always a
    // multiple of WRITE_CHUNK.
    const size_t total = size_gb << 30;

    std::printf("Generating %zu GiB of random 2-bit genotype data → %s\n",
                size_gb, path);

    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::perror("fopen"); return 1; }

    // mt19937_64 fills 8 bytes per call — fast enough not to bottleneck disk.
    std::mt19937_64 rng(0xDEAD'BEEF'CAFE'BABEull);

    // Buffer sized as uint64_t array; WRITE_CHUNK / 8 is always exact.
    std::vector<uint64_t> buf(WRITE_CHUNK / sizeof(uint64_t));

    const auto t_start = std::chrono::steady_clock::now();
    size_t written = 0;

    while (written < total) {
        const size_t chunk = std::min(WRITE_CHUNK, total - written);
        const size_t n64   = chunk / sizeof(uint64_t);
        for (size_t i = 0; i < n64; ++i)
            buf[i] = rng();

        // Handle any sub-8-byte tail (only possible if size_gb == 0, but
        // kept for robustness).
        const size_t tail = chunk % sizeof(uint64_t);
        if (tail) {
            const uint64_t last = rng();
            std::memcpy(reinterpret_cast<uint8_t*>(buf.data()) + n64 * sizeof(uint64_t),
                        &last, tail);
        }

        if (std::fwrite(buf.data(), 1, chunk, fp) != chunk) {
            std::perror("fwrite"); std::fclose(fp); return 1;
        }
        written += chunk;

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        std::printf("\r  %.2f / %.2f GiB  (%.2f GiB/s)",
                    written / (double)(1ULL << 30),
                    total   / (double)(1ULL << 30),
                    (written / (double)(1ULL << 30)) / elapsed);
        std::fflush(stdout);
    }

    std::fclose(fp);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::printf("\nDone: %.2f GiB in %.1f s  (%.2f GiB/s)\n",
                total / (double)(1ULL << 30), elapsed,
                (total / (double)(1ULL << 30)) / elapsed);
    return 0;
}
