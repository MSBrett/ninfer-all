# Build NInfer on Windows

Native Windows builds run the same engine as Linux: one model per server with concurrent
requests, the context cache, speculative decoding, Vision, and the OpenAI, Anthropic and llama.cpp
APIs, and pipeline stages over several GPUs (`--devices`), which exchange their boundary rows
through pinned host memory under WDDM ([pipeline stages](maintainer/pipeline-stages.md)). The
README inside a release archive is [`release-archive-windows.md`](release-archive-windows.md);
this guide is for a checkout.

**Status.** The last full native build and test run was in October 2026: every target of a `75`
build and of an `86` build with Visual Studio 2022 (MSVC 14.44) and CUDA 12.8, the test suite on a
TITAN RTX (TCC, and WDDM for the virtual-memory tests) and an RTX 3060 (WDDM), run by a contributor.
This repository has no Windows test host. CI compiles the host code of every translation unit with
MSVC 14.44 for `70;75;80;86;89` (CUDA 12.9) and `75;80;86;89;120a` (CUDA 13.4) on each change
to the build inputs (`scripts/check-host-compile.py`), but that host check builds no device code.
The separate `Native release` workflow compiles
and links the release products for two CUDA versions, checks their device images and tests `--help`
without the Toolkit on `PATH`. It has no GPU and does not qualify inference or performance.
The October 22 [dual-CUDA release run](https://github.com/iamwavecut/ninfer-all/actions/runs/39741764110)
completed native linking, device-image checks, startup checks, archive verification and upload
for both variants.
The optional DirectStorage reader and disk-KV bridge were qualified on Windows on October 25,
2026 with MSVC 14.44, the Windows 10.0.18362 SDK and DirectStorage 1.3.0. Enabled and disabled
host-only builds passed the storage tests, including real file reads, concurrent writers, CRC
failures and timeout cleanup. The complete `ninfer_core` static target also built for `sm_89`
with CUDA 12.8. This does not qualify GPU inference or establish a throughput gain.

MSVC builds delay-load `nvcuda.dll`, so `--help` can run without an NVIDIA driver.
GPU execution still requires a compatible driver and device.

Users run `120a` builds from `scripts\build-native.bat` with Visual Studio 2026 Build Tools and
CUDA 13.4 on an RTX 5070 Ti, and `89` builds with CUDA 13.4 on RTX 4090s
([field reports](performance/field-reports.md)).

## Requirements

- Windows 11 x64 and a recent NVIDIA driver;
- Visual Studio 2022 (MSVC v143) with CUDA 12.8 or newer for `75`, `86` and `89`, or Visual Studio
  2026 with CUDA 13.1 or newer for `120a`;
- CMake, Ninja and [vcpkg](https://github.com/microsoft/vcpkg), which provides FFmpeg and curl
  through `vcpkg.json`;
- the Microsoft Visual C++ runtime to run the executables, and a `.ninfer` artifact.

CUDA 12.8 refuses MSVC v145 (`C1189: unsupported Microsoft Visual Studio version!`), the compiler
of Visual Studio 2026, and `nvcc` takes whatever `cl.exe` comes first on `PATH`. With both Visual
Studio versions installed, build `86` and `89` from the VS 2022 environment (its `vcvars64.bat`
imports the compiler together with `INCLUDE` and `LIB`; adding only the compiler's directory to
`PATH` leaves the standard headers missing).

## Build

`scripts\build.ps1` imports the VS 2022 BuildTools environment itself and uses the Ninja
generator. It accepts an explicit `CUDACXX` or `CUDA_PATH`, otherwise selecting the installed
CUDA 12.8 or 12.9 toolchain, and builds into `build-ninja`:

```powershell
.\scripts\build.ps1                       # configure and build
.\scripts\build.ps1 -Test                 # then run the test suite
.\scripts\build.ps1 -Target ninfer-serve  # one target
.\scripts\build.ps1 -Arch 89              # another architecture (75, 80, 86 or 89)
.\scripts\build.ps1 -Arch 89 -DirectStorage -Package -Jobs 2 # optional SDK, portable archive, two build jobs
```

The default build uses the generator's parallelism. Use `-Jobs 2` to limit compiler and
assembler processes during an interactive desktop session. `-DirectStorage` enables the
optional SDK path; `-Test` runs tests only when explicitly requested.

The same build by hand:

```powershell
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --parallel
```

`scripts\build-native.bat` builds against a prebuilt vcpkg triplet tree instead of the vcpkg
toolchain, with Visual Studio 2026 and CUDA 13 by default, for `120a`:
`scripts\build-native.bat configure`, then `build`, or `target ninfer-serve` for one target. It
exports `VCPKG_ROOT` and `VCPKG_TARGET_TRIPLET`, from which `cmake\FindFFMPEG.cmake` finds FFmpeg,
and stages the triplet's DLLs beside every executable; `NINFER_VCVARS`, `NINFER_GENERATOR`,
`NINFER_CUDA_PATH`, `NINFER_VCPKG_ROOT`, `NINFER_VCPKG_TRIPLET`, `NINFER_CUDA_ARCH` and
`NINFER_BUILD_DIR` override its defaults.

A standalone `.cu` probe outside the build tree can be compiled with `nvcc` directly from
PowerShell (Git Bash rewrites MSVC-style flags into paths). Keep probe builds out of the
repository root: `nvcc` writes `.exp` and `.lib` files next to the output.

Windows builds differ from Linux builds in three places: they decode PNG images natively (a
prebuilt FFmpeg tree may have no PNG decoder), they pass the NVFP4 TMA descriptors through device
memory, since MSVC cannot pass them by value, and WDDM charges pinned host memory against the
card's memory. Every Windows build offers `--wddm-evictable-budget`, which sizes the KV cache against
the card's memory less the weights and a 512 MiB desktop floor instead of what is free at startup,
and commits the device arenas at startup. Another process that later takes memory makes WDDM page
some of this process's memory out, which slows generation until it comes back. The Windows-only
build option:

| option | effect |
|---|---|
| `NINFER_DIRECTSTORAGE=ON` | fetches the pinned DirectStorage 1.3.0 SDK/runtime (latest stable, checked October 25, 2026) and offers `--disk-kv-directstorage` for disk-tier restores |

The other build options are in the [Linux guide](build-linux.md#build-options).

### Remaining DirectStorage work

The host reader, disk-KV bridge and `sm_89` Core build are qualified as described above.
The October 25 development batch also fixes SDK discovery beside a portable backend DLL,
adds Windows CI coverage, and exposes `-DirectStorage` and `-Jobs` in the Windows build script.
The backend-loading probe failed before that fix; the completed batch still needs verification.
The following work remains before claiming a qualified inference path or a performance benefit.
Keep `NINFER_DIRECTSTORAGE` optional and off by default. SDK and queue behavior are described in
[serving](serving.md); host-test commands are in [tests](../tests/README.md#build-and-run).

1. **Build and package the complete Windows product.** Build and link the server and shared
   runtime for `sm_89` with DirectStorage enabled, using the supported release toolchains.
   Produce and extract a real release archive. Completion requires both SDK DLLs and
   `DirectStorage-LICENSE.txt` in each enabled backend, successful startup with development
   directories removed from `PATH`, and a clear error when DirectStorage is requested without
   its DLLs. The Core archive build, staging fixtures and package-script syntax check already
   passed; they do not establish a complete product link or archive result.

2. **Qualify real Engine restores on a Windows RTX 4090.** Use the latest official Qwen3.8-27B
   `.ninfer` artifact (the engine's Qwen3.5-core architecture) and `--use-original-prefix-caching --disk-kv-restore --disk-kv-path PATH`.
   Populate the disk tier, restart with no resident prefix, then compare the same persisted
   pages with and without `--disk-kv-directstorage`. Completion requires proof of actual
   DirectStorage use, byte-identical restored KV/state and matching deterministic continuation
   against the ordinary disk-reader baseline. Cover single and concurrent requests, and the
   speculative backend intended for release. Use port `8899` for this Windows qualification.
   The current task uses the local RTX 4090; confirm it is free before execution and leave
   other applications alone. Remote qualification uses the project's
   [rented-host workflow](../scripts/pods/README.md); Linux-only hosts cannot qualify this
   Windows path. Record unavailable hardware or artifacts as prerequisites.

3. **Exercise failure recovery through Engine.** Extend that workload with missing and corrupt
   pages, failed reads and timeout cleanup while cache writers are active. Completion requires
   safe fallback or prompt recomputation, released page claims, reusable destinations and a
   successful later request. Record queue-close latency under failure: the 10-second wait
   deadline does not bound how long issued I/O takes to settle. Existing host tests establish
   reader/bridge behavior, not complete Engine recovery.

4. **Measure whether DirectStorage helps.** Compare the enabled runtime option against the
   existing eight-reader positioned-I/O path in the same build, with the same artifact, prompts,
   concurrency, cache contents, NVMe volume, KV format and speculative settings. Separate cold
   and warm OS-cache conditions. Report restored bytes, restore time, time to first token, CPU
   cost and writer/request stalls, including the distribution and worst regressions. The
   per-batch mapping/handle changes have a cost; do not infer an inference gain from API tests
   or an I/O-only result. Completion is a representative result and an explicit disposition
   of any regressions, not a selected best case.

5. **Make regression coverage repeatable.** Windows CI now has enabled/disabled host storage
   runs, including the real SDK and the shared-backend loading regression, on the first shard
   of each CUDA configuration. Those jobs still need a recorded passing run. Build the new
   host-only storage target and run its disabled-path tests on Linux to check the Core target
   split and ordinary file-I/O behavior. Completion requires recorded passing jobs; neither
   Linux compilation nor automated Windows coverage was established by the local checks.

6. **Resolve the SDK warning follow-up.** The tested Windows 10.0.18362 headers emit C5105.
   Repeat the affected build with a newer Windows SDK compatible with the selected CUDA/MSVC
   pair and record whether the warning remains. Do not suppress it without establishing its
   cause. This diagnostic follow-up is separate from the passing storage correctness tests.

## Run

`scripts\download-model.bat qwen38-27b` fetches the pinned official Qwen3.8-27B artifact and checks
its size and SHA-256 (`qwen36-27b` and `qwen36-35b-a3b` work the same way), and
`scripts\run.bat qwen38-27b` serves it on `127.0.0.1:8899` with the measured profile; the profiles and their overrides
are in [launcher profiles](maintainer/launcher-profiles.md). Any other artifact starts with
`ninfer-serve.exe` and the flags the [launch recipes](recipes.md) give for it, for example:

```powershell
.\build-ninja\apps\ninfer-serve.exe models\qwen3_8_27b.ninfer --model-id qwen3.8-27b `
  --port 8899 --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4 --gdn-state-fp16 `
  --spec mtp --draft-tokens 3
```

`--kv-capacity auto` sizes the KV pool from the memory left after startup but keeps 1 GiB free;
on a 24 GB card that serves a desktop, an explicit capacity is easier to predict. The
[HTTP serving guide](serving.md) and `ninfer-serve.exe --help` cover every option.

## Release archive

The `Native release` workflow builds two single-CUDA x64 packages with VS 2022 and Ninja, then assembles a universal package:

| Toolkit | Device images available to each executable |
|---|---|
| CUDA 12.9.2 | `70;75;80;86;89` (Volta through Ada) |
| CUDA 13.4.3 | `75;80;86;89;120a` (Turing through consumer Blackwell) |

Each Toolkit compiles its release objects in eight jobs. Every CUDA object retains the complete
architecture set above. Jobs checkpoint separate HF cache families; a final job restores them,
links the executables and verifies the package. At most eight object jobs run together across
both Toolkits. Each runner uses two compiler processes with two nvcc optimization threads each,
to limit the memory pressure observed on the four-CPU Windows runners. This does not establish
a bound on compilation time for the largest individual source files.

Windows release builds set `NINFER_CUDA_PRUNE_PTX=ON`. NVIDIA's `nvprune` removes PTX from
the CUDA static archives before linking, retaining machine code for every listed GPU architecture.
This reduces the image size without recompiling cached objects. These packages do not provide
PTX fallback for GPUs that cannot run any of their machine-code images. Local builds retain PTX
by default; enabling this option requires the Toolkit's `nvprune` component.

The workflow also sets `NINFER_WINDOWS_KERNEL_DLL=ON`: GGML kernels reside in
`ninfer_ggml_quants.dll` inside the selected backend directory. The DLL carries the same architecture set and
retains its PTX; the other CUDA archives are pruned. This separates device code across PE images
to address MSVC's 2 GiB limit without removing GPU targets. The DLL uses the existing function-only
GGML bridge and is staged for apps, the product benchmark and tests. Local builds default to the
static library. The packager checks device images in the kernel DLL and the shared runtime DLL.

Every push to `master` starts the current build and cancels the superseded run.
Feature branches build only through a manual run, which builds the packages
without publication by default. Select `publish` on `master` to publish. Both Windows and both Linux jobs must
pass before publication. The workflow creates an source-identified `YYYY-MM-DD.N` release,
then marks it **latest** after checking every uploaded asset. It leaves an older latest release
available if compilation, packaging or upload fails. See [release process](maintainer/release-process.md).

For a local build, `scripts\package-release.ps1` (or `.\scripts\build.ps1 -Package`) derives the
Toolkit and device images from `CMakeCache.txt`. It requires `cuobjdump` in that Toolkit and `7z`
on `PATH` and `NINFER_SHARED_RUNTIME=ON` in the build. It packages all six command EXEs, their runtime DLLs,
licences and launchers into `.7z.001` volumes of at most 1900 MiB. Download every volume of one
package into the same folder and extract the first with 7-Zip. Keep the backend subdirectories intact.

`BUILD-INFO.json` records the version, source revision, source modifications, CUDA version and device
images. Archive names use the platform, CUDA version, GPU architectures and 12-character source commit,
for example `ninfer-windows-x64-cuda13.4-sm75-80-86-89-120a-0123456789ab.7z.001`.

The previous static-EXE layout produced CUDA 13.4 executables of about 1.95 GB each, plus a 313 MB
GGML DLL. Its package was 6.73 GB unpacked and 6.37 GB archived. The shared-runtime layout removes
this duplication between commands. Its actual linked sizes and GPU behavior still require native
validation. The packager rejects device modules at or above 2 GiB. Archive splitting only addresses
GitHub's asset limit.

The universal name is `ninfer-windows-x64-cuda12.9-13.4-<commit12>`. Each small EXE chooses one backend
from `backends/cuda12/` and `backends/cuda13/`. Use `--cuda-backend auto|12|13` to override selection.
`--help` and `--version` need no driver. See the [release process](maintainer/release-process.md)
for selection, cache and verification details. A successful startup check does not establish GPU
correctness or throughput.
