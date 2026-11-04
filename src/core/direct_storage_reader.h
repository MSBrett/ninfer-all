#pragma once

// Reads file ranges into host memory through Microsoft DirectStorage, on Windows builds with
// NINFER_DIRECTSTORAGE: every range of a batch goes to the NVMe queue at once, with little CPU
// work per read. Other builds have no reader, and callers keep their mapped-file reads.

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
    std::size_t file = 0; // index in open_files(); zero for read(path, ranges)
};

class DirectStorageReader {
public:
    [[nodiscard]] static bool available_in_build() noexcept;

    // Throws when this build or machine has no working DirectStorage.
    [[nodiscard]] static std::unique_ptr<DirectStorageReader>
    open(std::chrono::milliseconds batch_timeout = std::chrono::seconds(10));

    // Immutable files stay open for the reader's lifetime, sharing one persistent queue.
    // Unlike open(), this must not be used for a writable/mapped prefix-cache file.
    [[nodiscard]] static std::unique_ptr<DirectStorageReader>
    open_files(std::span<const std::string> paths,
               std::chrono::milliseconds batch_timeout = std::chrono::seconds(10));

    virtual ~DirectStorageReader() = default;

    // Reads every range of the file at `path` (UTF-8); false when any read failed.
    [[nodiscard]] virtual bool read(const std::string& path,
                                    std::span<const FileRangeRead> reads) = 0;

    // Batched ranges from open_files(), with the same completion/lifetime guarantee.
    [[nodiscard]] virtual bool read(std::span<const FileRangeRead> reads) = 0;
};

} // namespace ninfer
