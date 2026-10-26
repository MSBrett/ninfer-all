#pragma once

// Session statistics panel pinned beneath the scrolling operational console log. It aggregates the
// same completed-request measurements the `req#N done` records print, over the whole session and
// over the most recent requests, and redraws product::TerminalPanel after every change.

#include "serve/request_events.h"
#include "serve/load_report.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::product {
class TerminalPanel;
}

namespace ninfer::serve {

// One completed request's contribution. Speculative totals from the Engine include n-gram rounds;
// the model fields here are the remainder (MTP or DFlash drafting), so the two sources never double
// count.
struct ConsoleRequestSample {
    std::uint64_t request_id = 0;
    std::uint64_t completion_tokens = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    double ttft_seconds                    = 0.0;
    std::uint64_t prompt_tokens            = 0;
    std::uint64_t cache_hit_tokens         = 0;
    std::uint64_t computed_prefill_tokens  = 0;
    double prefill_seconds                 = 0.0;
    std::uint64_t decode_tokens            = 0;
    double decode_seconds                  = 0.0;
    std::uint64_t model_rounds             = 0;
    std::uint64_t model_drafted_tokens     = 0;
    std::uint64_t model_accepted_tokens    = 0;
    std::uint64_t ngram_rounds             = 0;
    std::uint64_t ngram_drafted_tokens     = 0;
    std::uint64_t ngram_accepted_tokens    = 0;
    std::uint64_t archive_drafted_tokens   = 0;
    std::uint64_t archive_accepted_tokens  = 0;
};

[[nodiscard]] ConsoleRequestSample make_console_request_sample(const GenerationOutcome& outcome);

// Sums over a set of completed requests. Rates and ratios are formed from the sums, so each request
// weighs by its tokens and time rather than counting as one equal vote.
struct ConsoleStatsTotals {
    std::uint64_t requests = 0;
    ConsoleRequestSample sum;

    void add(const ConsoleRequestSample& sample) noexcept;
};

struct ConsoleStatsSnapshot {
    std::uint64_t completed = 0;
    std::uint64_t failed    = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t rejected  = 0;
    std::optional<std::uint32_t> running;
    std::optional<std::uint32_t> waiting;
    std::size_t recent_window = 0;
    // Model drafter that produced the speculative columns; None until a request used one.
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ConsoleStatsTotals session;
    ConsoleStatsTotals recent;
    std::vector<ConsoleRequestSample> recent_requests;
    struct Model {
        LoadCapacity capacity;
        ninfer::RuntimeStats stats;
        std::size_t admitted = 0;
        std::size_t peak_admitted = 0;
    };
    // One snapshot per loaded model. Replaced atomically each tick, including an empty router.
    std::vector<Model> models;
};

// Pure rendering of the panel rows; the terminal fits them to its width.
[[nodiscard]] std::vector<std::string>
render_console_stats_panel(const ConsoleStatsSnapshot& snapshot);

// The dashboard adapts to the available terminal, reserving space for the scrolling log.
[[nodiscard]] std::vector<std::string> render_tui_stats_panel(
    const ConsoleStatsSnapshot& snapshot, std::size_t columns, std::size_t rows, bool colours);

class ConsoleStatsPanel {
public:
    static constexpr std::size_t kRecentRequests = 10;

    explicit ConsoleStatsPanel(std::shared_ptr<product::TerminalPanel> panel, bool tui = false);

    void request_done(const GenerationOutcome& outcome, std::uint64_t request_id = 0);
    void request_failure(const RequestFailure& failure);
    void request_rejected(const RequestFailure& failure);
    void runtime(const ninfer::RuntimeStats& current);
    void runtime(std::vector<ConsoleStatsSnapshot::Model> models);
    void show();

    [[nodiscard]] ConsoleStatsSnapshot snapshot() const;

private:
    void publish_locked();

    std::shared_ptr<product::TerminalPanel> panel_;
    mutable std::mutex mutex_;
    ConsoleStatsSnapshot state_;
    std::deque<ConsoleRequestSample> recent_;
    bool tui_ = false;
};

} // namespace ninfer::serve
