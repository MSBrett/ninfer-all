#include "core/direct_storage_reader.h"

#include <stdexcept>

#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

#    include <dstorage.h>

#    include <algorithm>
#    include <chrono>
#    include <cstdio>
#    include <filesystem>
#    include <limits>
#    include <mutex>
#    include <type_traits>

namespace ninfer {
namespace {

template <typename Interface>
class ComReference {
public:
    ComReference()                               = default;
    ComReference(const ComReference&)            = delete;
    ComReference& operator=(const ComReference&) = delete;

    ComReference(ComReference&& other) noexcept : pointer_(other.pointer_) {
        other.pointer_ = nullptr;
    }

    ComReference& operator=(ComReference&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_       = other.pointer_;
            other.pointer_ = nullptr;
        }
        return *this;
    }

    ~ComReference() { reset(); }

    void reset() noexcept {
        if (pointer_ != nullptr) { pointer_->Release(); }
        pointer_ = nullptr;
    }

    [[nodiscard]] Interface* get() const noexcept { return pointer_; }

    Interface* operator->() const noexcept { return pointer_; }

    void** out() noexcept {
        reset();
        return reinterpret_cast<void**>(&pointer_);
    }

private:
    Interface* pointer_ = nullptr;
};

std::wstring widen(const std::string& text) {
    if (text.empty()) { return {}; }
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DirectStorage path is too long");
    }
    const int length =
        ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (length <= 0) { throw std::invalid_argument("DirectStorage path is not valid UTF-8"); }
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                          static_cast<int>(text.size()), wide.data(), length);
    return wide;
}

void require_sdk_success(HRESULT result, const char* operation) {
    if (SUCCEEDED(result)) { return; }
    char code[11]{};
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(result));
    throw std::runtime_error(std::string("DirectStorage ") + operation +
                             " failed (HRESULT " + code + ")");
}

HMODULE load_runtime_module() {
    HMODULE owner = nullptr;
    if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCWSTR>(&load_runtime_module), &owner)) {
        throw std::runtime_error("DirectStorage reader module could not be located");
    }
    std::wstring path(32768, L'\0');
    const DWORD length = ::GetModuleFileNameW(owner, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) {
        throw std::runtime_error("DirectStorage reader module path could not be read");
    }
    path.resize(length);
    const auto sdk = std::filesystem::path(path).parent_path() / L"dstorage.dll";
    auto module = ::LoadLibraryExW(sdk.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr) {
        throw std::runtime_error("dstorage.dll could not be loaded beside the DirectStorage reader "
                                 "(Windows error " + std::to_string(::GetLastError()) + ")");
    }
    return module;
}

class WindowsDirectStorageReader final : public DirectStorageReader {
public:
    explicit WindowsDirectStorageReader(std::chrono::milliseconds timeout) : timeout_(timeout) {
        if (timeout.count() < 0 || timeout.count() >= INFINITE) {
            throw std::invalid_argument("DirectStorage timeout is outside the Win32 wait range");
        }
        // Explicit loading keeps every executable usable without the optional SDK DLLs.
        module_.reset(load_runtime_module());
        using GetFactory = HRESULT(WINAPI*)(REFIID, void**);
        const auto get_factory = reinterpret_cast<GetFactory>(
            ::GetProcAddress(module_.get(), "DStorageGetFactory"));
        if (get_factory == nullptr) {
            const DWORD error = ::GetLastError();
            throw std::runtime_error("dstorage.dll does not export DStorageGetFactory "
                                     "(Windows error " + std::to_string(error) + ")");
        }
        require_sdk_success(get_factory(__uuidof(IDStorageFactory), factory_.out()),
                            "factory creation");
        completion_.reset(::CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (!completion_) { throw std::runtime_error("DirectStorage completion event failed"); }
        DSTORAGE_QUEUE_DESC queue{};
        queue.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queue.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
        queue.Priority   = DSTORAGE_PRIORITY_NORMAL;
        queue.Name       = "ninfer disk KV restore";
        require_sdk_success(factory_->CreateQueue(&queue, __uuidof(IDStorageQueue1), queue_.out()),
                            "queue creation");
        require_sdk_success(factory_->CreateStatusArray(1, "ninfer disk KV restore",
                                                        __uuidof(IDStorageStatusArray), status_.out()),
                            "status array creation");
    }

    ~WindowsDirectStorageReader() override {
        close_queue();
    }

    bool read(const std::string& path, std::span<const FileRangeRead> reads) override {
        std::lock_guard lock(mutex_);
        if (broken_) { return false; }
        if (reads.empty()) { return true; }
        ComReference<IDStorageFile> opened = open_file(path);
        IDStorageFile* file = opened.get();
        if (file == nullptr) { return false; }
        struct CloseFile {
            IDStorageFile* file;
            ~CloseFile() { file->Close(); }
        } close{file};
        BY_HANDLE_FILE_INFORMATION info{};
        if (FAILED(file->GetFileInformation(&info))) { return false; }
        const std::uint64_t file_size = (std::uint64_t{info.nFileSizeHigh} << 32U) |
                                        info.nFileSizeLow;
        // EnqueueRequest may submit automatically. Validate the entire call before any enqueue,
        // including later ranges, so a refused call leaves no untracked writes to borrowed memory.
        for (const FileRangeRead& range : reads) {
            if (range.offset > file_size || range.destination.size() > file_size - range.offset) {
                return false;
            }
        }
        constexpr std::size_t kBatchCapacity = DSTORAGE_MAX_QUEUE_CAPACITY - 2U;
        constexpr std::size_t kRequestBytes  = 32U * 1024U * 1024U;
        std::size_t range_index = 0;
        std::size_t range_used  = 0;
        while (range_index < reads.size()) {
            const auto deadline = std::chrono::steady_clock::now() + timeout_;
            if (!::ResetEvent(completion_.get())) { return false; }
            std::size_t count = 0;
            while (range_index < reads.size() && count < kBatchCapacity) {
                const FileRangeRead& range = reads[range_index];
                if (range_used == range.destination.size()) {
                    ++range_index;
                    range_used = 0;
                    continue;
                }
                const auto bytes = static_cast<UINT32>(
                    std::min(kRequestBytes, range.destination.size() - range_used));
                DSTORAGE_REQUEST request{};
                request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                request.Source.File.Source        = file;
                request.Source.File.Offset        = range.offset + range_used;
                request.Source.File.Size          = bytes;
                request.Destination.Memory.Buffer = range.destination.data() + range_used;
                request.Destination.Memory.Size   = bytes;
                request.UncompressedSize          = bytes;
                request.CancellationTag           = batch_tag_;
                queue_->EnqueueRequest(&request);
                range_used += bytes;
                ++count;
            }
            if (count == 0) { break; }
            queue_->EnqueueStatus(status_.get(), 0);
            queue_->EnqueueSetEvent(completion_.get());
            queue_->Submit();
            if (!wait_for_batch(deadline)) { return false; }
            ++batch_tag_;
        }
        return true;
    }

private:
    ComReference<IDStorageFile> open_file(const std::string& path) {
        std::wstring wide;
        try {
            wide = widen(path);
        } catch (const std::invalid_argument&) { return {}; }
        ComReference<IDStorageFile> file;
        if (FAILED(factory_->OpenFile(wide.c_str(), __uuidof(IDStorageFile), file.out()))) {
            return {};
        }
        return file;
    }

    void close_queue() noexcept {
        if (queue_.get() != nullptr) {
            // The SDK guarantees no request can complete after Close returns. In particular,
            // cancellation alone does not give the caller permission to reuse its destination.
            queue_->Close();
            queue_.reset();
        }
    }

    bool wait_for_batch(std::chrono::steady_clock::time_point deadline) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        const auto millis = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
        if (remaining <= std::chrono::steady_clock::duration::zero() ||
            ::WaitForSingleObject(completion_.get(), static_cast<DWORD>(millis)) != WAIT_OBJECT_0) {
            queue_->CancelRequestsWithTag(~UINT64{0}, batch_tag_);
            broken_ = true;
            close_queue();
            return false;
        }
        if (FAILED(status_->GetHResult(0))) {
            DSTORAGE_ERROR_RECORD record{};
            queue_->RetrieveErrorRecord(&record);
            return false;
        }
        return true;
    }

    // Destroy the queue and its completion objects before unloading their implementation DLL.
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&::FreeLibrary)>
        module_{nullptr, &::FreeLibrary};
    std::unique_ptr<void, decltype(&::CloseHandle)> completion_{nullptr, &::CloseHandle};
    std::chrono::milliseconds timeout_;
    std::mutex mutex_;
    UINT64 batch_tag_ = 1;
    bool broken_      = false;
    ComReference<IDStorageFactory> factory_;
    ComReference<IDStorageQueue1> queue_;
    ComReference<IDStorageStatusArray> status_;
};

} // namespace

std::unique_ptr<DirectStorageReader> DirectStorageReader::open(std::chrono::milliseconds timeout) {
    return std::make_unique<WindowsDirectStorageReader>(timeout);
}

} // namespace ninfer

#else

namespace ninfer {

std::unique_ptr<DirectStorageReader> DirectStorageReader::open(std::chrono::milliseconds timeout) {
    (void)timeout;
    throw std::runtime_error("this build has no DirectStorage (Windows with NINFER_DIRECTSTORAGE)");
}

} // namespace ninfer

#endif
