#pragma once

// Reads file ranges into host memory through Microsoft DirectStorage, on Windows builds with
// NINFER_DIRECTSTORAGE: ranges are submitted together through a persistent SDK queue.
// Other builds have no reader, and callers keep their ordinary file reads.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace ninfer {

struct FileRangeRead {
    std::uint64_t offset = 0;
    std::span<std::byte> destination;
};

class DirectStorageReader {
public:
    [[nodiscard]] static constexpr bool available_in_build() noexcept {
#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
        return true;
#else
        return false;
#endif
    }

    // Throws when this build or machine has no working DirectStorage.
    [[nodiscard]] static std::unique_ptr<DirectStorageReader>
    open(std::chrono::milliseconds batch_timeout = std::chrono::seconds(10));

    virtual ~DirectStorageReader() = default;

    // Reads every range of the file at `path` (UTF-8); false when any read failed.
    // No request can access a destination after this returns, including after a timeout.
    // A timeout closes the queue before fallback; closing may wait for issued I/O to settle.
    [[nodiscard]] virtual bool read(const std::string& path,
                                    std::span<const FileRangeRead> reads) = 0;
};

} // namespace ninfer
