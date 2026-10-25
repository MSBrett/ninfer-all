#include "core/direct_storage_reader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer_directstorage_" + std::to_string(std::random_device{}()));
    TemporaryDirectory() {
        if (!std::filesystem::create_directory(path)) {
            throw std::runtime_error("temporary directory already exists");
        }
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void startup_without_runtime(const std::filesystem::path& root, const char* executable) {
    const auto isolated = root / "without-runtime";
    std::filesystem::create_directory(isolated);
    const auto copy = isolated / "probe.exe";
    std::filesystem::copy_file(std::filesystem::absolute(executable), copy);
    std::wstring command = L"\"" + copy.wstring() + L"\" --startup-probe";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!::CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                          nullptr, isolated.c_str(), &startup, &child)) {
        throw std::runtime_error("cannot start the isolated DirectStorage startup probe");
    }
    ::CloseHandle(child.hThread);
    const DWORD waited = ::WaitForSingleObject(child.hProcess, 10000);
    DWORD exit = 1;
    if (waited == WAIT_OBJECT_0) { ::GetExitCodeProcess(child.hProcess, &exit); }
    else { ::TerminateProcess(child.hProcess, 1); }
    ::CloseHandle(child.hProcess);
    check(waited == WAIT_OBJECT_0 && exit == 0,
          "a DirectStorage-enabled executable starts without SDK DLLs and refuses an unavailable reader");
}

void exercise_reader(const std::filesystem::path& root) {
    // Unaligned offsets and a range beyond the SDK default staging buffer exercise real reads.
    const auto file = root / std::filesystem::path(u8"pages_\u03bb.bin");
    const auto utf8_path = file.u8string();
    const std::string path(reinterpret_cast<const char*>(utf8_path.data()), utf8_path.size());
    std::vector<std::byte> source(32U * 1024U * 1024U + 4097U);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::byte>((i * 31U + i / 97U) & 255U);
    }
    {
        std::ofstream output(file, std::ios::binary);
        output.write(reinterpret_cast<const char*>(source.data()), source.size());
        if (!output) { throw std::runtime_error("cannot write DirectStorage fixture"); }
    }
    auto reader = ninfer::DirectStorageReader::open();
    check(reader->read(path + ".missing", {}), "an empty read has no file dependency");
    std::vector<std::byte> out(source.size());
    const ninfer::FileRangeRead full{.offset = 0, .destination = out};
    check(reader->read(path, std::span(&full, 1)) && out == source,
          "a range over 32 MiB is split and restores exactly");

    std::vector<std::byte> first(97), second(511), third(32768);
    const ninfer::FileRangeRead ranges[] = {
        {.offset = 149, .destination = first}, {.offset = 7, .destination = second},
        {.offset = 4144, .destination = third}, {.offset = source.size(), .destination = {}}};
    check(reader->read(path, ranges) &&
              std::equal(first.begin(), first.end(), source.begin() + 149) &&
              std::equal(second.begin(), second.end(), source.begin() + 7) &&
              std::equal(third.begin(), third.end(), source.begin() + 4144),
          "unaligned, unordered, and empty ranges preserve each destination");

    // More requests than one SDK queue can hold, independently checked against source bytes.
    constexpr std::size_t count = 8193;
    std::vector<std::byte> small(count * 17);
    std::vector<ninfer::FileRangeRead> many;
    many.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        many.push_back({.offset = i * 37, .destination = std::span(small).subspan(i * 17, 17)});
    }
    bool exact = reader->read(path, many);
    for (std::size_t i = 0; i < count; ++i) {
        exact = exact && std::equal(small.begin() + i * 17, small.begin() + (i + 1) * 17,
                                    source.begin() + i * 37);
    }
    check(exact, "a read larger than queue capacity completes in ordered batches");

    const auto canary = static_cast<std::byte>(0x5a);
    std::fill(first.begin(), first.end(), canary);
    const ninfer::FileRangeRead invalid[] = {
        {.offset = 0, .destination = first},
        {.offset = std::numeric_limits<std::uint64_t>::max(), .destination = second}};
    check(!reader->read(path, invalid), "a later overflowing file range refuses the entire call");
    const ninfer::FileRangeRead valid{.offset = 7, .destination = second};
    check(reader->read(path, std::span(&valid, 1)) &&
              std::all_of(first.begin(), first.end(), [&](std::byte b) { return b == canary; }),
          "invalid input leaves no earlier queued writes to caller memory");
    const ninfer::FileRangeRead eof{.offset = source.size(), .destination = first};
    check(!reader->read(path, std::span(&eof, 1)), "a range past EOF is refused");
    check(!reader->read(path + ".missing", std::span(&valid, 1)), "a missing file fails");
    check(!reader->read(std::string(1, static_cast<char>(0xff)), std::span(&valid, 1)),
          "an invalid UTF-8 path fails without throwing");

    std::atomic<bool> concurrent_ok{true};
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < 4; ++i) {
        threads.emplace_back([&, i] {
            std::vector<std::byte> part(65536);
            const ninfer::FileRangeRead range{.offset = i * 65536, .destination = part};
            if (!reader->read(path, std::span(&range, 1)) ||
                !std::equal(part.begin(), part.end(), source.begin() + i * 65536)) {
                concurrent_ok.store(false);
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    check(concurrent_ok.load(), "concurrent callers get independent exact results");

    auto timed = ninfer::DirectStorageReader::open(std::chrono::milliseconds(0));
    check(!timed->read(path, std::span(&full, 1)), "an expired batch deadline closes the queue");
    std::fill(out.begin(), out.end(), canary);
    check(!timed->read(path, std::span(&full, 1)), "a closed reader accepts no more work");
    std::vector<std::byte> verify(source.size());
    const ninfer::FileRangeRead after_timeout{.offset = 0, .destination = verify};
    check(reader->read(path, std::span(&after_timeout, 1)) && verify == source &&
              std::all_of(out.begin(), out.end(), [&](std::byte b) { return b == canary; }),
          "timed-out I/O cannot modify destinations after fallback may reuse them");
    timed.reset();
    reader.reset();
    check(std::filesystem::remove(file), "destroying readers releases the file handles");
}
#endif
} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
    if (argc == 3 && std::string(argv[1]) == "--backend-probe") {
        // Match the portable launcher's backend loading, with no developer or executable
        // directories available to satisfy the reader's later SDK load.
        if (!::SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)) { return 1; }
        const auto path = std::filesystem::absolute(argv[2]).wstring();
        HMODULE backend = ::LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (backend == nullptr) { return 1; }
        using Probe = int (*)();
        const auto probe = reinterpret_cast<Probe>(
            ::GetProcAddress(backend, "ninfer_directstorage_backend_probe"));
        const int result = probe != nullptr ? probe() : 1;
        ::FreeLibrary(backend);
        return result;
    }
#endif

#ifdef _WIN32
    if (argc == 2 && std::string(argv[1]) == "--startup-probe") {
        if (::GetModuleHandleW(L"dstorage.dll") != nullptr ||
            !::SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_APPLICATION_DIR)) { return 1; }
        try { auto reader = ninfer::DirectStorageReader::open(); }
        catch (const std::runtime_error&) { return 0; }
        return 1;
    }
#endif
    try {
        if (!ninfer::DirectStorageReader::available_in_build()) {
            bool refused = false;
            try { auto reader = ninfer::DirectStorageReader::open(); }
            catch (const std::runtime_error&) { refused = true; }
            check(refused, "a build without DirectStorage refuses the reader explicitly");
        }
#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
        else {
            TemporaryDirectory root;
            startup_without_runtime(root.path, argv[0]);
            exercise_reader(root.path);
        }
#endif
    } catch (const std::exception& error) {
        std::cerr << "DirectStorage test: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    if (ninfer::DirectStorageReader::available_in_build()) {
        std::cout << "DirectStorage reader: exact ranges, queue boundaries, failures and lifetime ok\n";
    } else {
        std::cout << "DirectStorage disabled: explicit refusal ok\n";
    }
    return 0;
}
