#include "serve/console_stats.h"

#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string_view>

namespace ninfer::serve {
namespace {

using product::format_pretty_count;
using product::format_pretty_duration;
using product::format_pretty_percent;
using product::format_pretty_rate;

constexpr std::string_view kReset = "\x1b[0m";
constexpr std::string_view kAccent = "\x1b[96m";
constexpr std::string_view kMuted = "\x1b[37m";
constexpr std::string_view kValue = "\x1b[1m";
constexpr std::string_view kWarning = "\x1b[93m";

std::string count(std::uint64_t value) { return format_pretty_count(value); }

std::string percent(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? "--" : format_pretty_percent(static_cast<double>(part) / whole);
}

std::string rate(std::uint64_t tokens, double seconds) {
    return seconds > 0.0 ? format_pretty_rate(static_cast<double>(tokens) / seconds, "tok") : "--";
}

std::string gib(std::uint64_t bytes) {
    char text[40];
    std::snprintf(text, sizeof(text), "%.2f", static_cast<double>(bytes) / (1ULL << 30));
    return text;
}

// Nearest-rank percentiles over actual recent completed requests, excluding phases that did not
// execute. Unlike the log table's ratios of sums these give each represented request one vote.
template <typename Value>
std::optional<double> percentile(const ConsoleStatsSnapshot& snapshot, double fraction, Value value) {
    std::vector<double> values;
    for (const auto& sample : snapshot.recent_requests) {
        if (const auto item = value(sample); item && std::isfinite(*item)) { values.push_back(*item); }
    }
    if (values.empty()) { return std::nullopt; }
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1];
}

std::string quantile_rate(const ConsoleStatsSnapshot& state, double fraction, bool decode) {
    const auto result = percentile(state, fraction, [decode](const ConsoleRequestSample& s)
        -> std::optional<double> {
        const double seconds = decode ? s.decode_seconds : s.prefill_seconds;
        if (seconds <= 0.0) { return std::nullopt; }
        return static_cast<double>(decode ? s.decode_tokens : s.computed_prefill_tokens) / seconds;
    });
    return result ? format_pretty_rate(*result, "tok") : "--";
}

std::string quantile_ttft(const ConsoleStatsSnapshot& state, double fraction) {
    const auto result = percentile(state, fraction, [](const ConsoleRequestSample& s)
        -> std::optional<double> { return s.ttft_seconds; });
    return result ? format_pretty_duration(*result) : "--";
}

std::string pad(std::string_view text, std::size_t width) {
    std::string fitted = product::fit_terminal_line(text, width);
    fitted.append(width - product::terminal_display_width(fitted), ' ');
    return fitted;
}

std::string cells(const std::vector<std::string>& text, std::size_t width) {
    const std::size_t cell_width = width / text.size();
    std::string line;
    for (std::size_t index = 0; index < text.size(); ++index) {
        line += pad(text[index], index + 1 == text.size() ? width - cell_width * index : cell_width);
    }
    return line;
}

// ASCII bars and borders stay aligned on Windows code pages and terminals without Unicode fonts.
std::string bar(std::uint64_t used, std::uint64_t capacity, std::size_t width) {
    if (capacity == 0) { return "[" + std::string(width, '-') + "] --"; }
    const double ratio = static_cast<double>(used) / capacity;
    const auto filled = static_cast<std::size_t>(std::clamp(ratio, 0.0, 1.0) * width);
    return "[" + std::string(filled, '=') + std::string(width - filled, ' ') + "] " +
           format_pretty_percent(ratio);
}

struct Live {
    std::uint64_t running = 0, concurrency = 0, prefill = 0, decode = 0, waiting = 0;
    std::uint64_t materializing = 0, capturing = 0, terminal = 0;
    std::uint64_t admitted = 0, peak = 0, kv_used = 0, kv_capacity = 0;
    std::uint64_t host_used = 0, host_capacity = 0, device_state = 0, host_state = 0;
    std::uint64_t spill = 0, evict = 0, demote = 0, degrade = 0, fallback = 0;
    std::uint64_t down = 0, up = 0, prefill_tokens = 0, decode_tokens = 0;
    double prefill_seconds = 0.0, decode_seconds = 0.0;
};

Live live_totals(const ConsoleStatsSnapshot& state) {
    Live out;
    for (const auto& model : state.models) {
        const auto& s = model.stats;
        const auto& c = model.capacity;
        out.running += s.running_requests;
        out.concurrency += c.max_concurrency;
        out.prefill += s.prefilling_requests;
        out.decode += s.decode_ready_requests;
        out.waiting += s.waiting_requests;
        out.materializing += s.materializing_requests;
        out.capturing += s.capture_pending_requests;
        out.terminal += s.terminal_pending_requests;
        out.admitted += model.admitted;
        out.peak += model.peak_admitted;
        out.kv_used += s.device_main_kv_occupied_pages;
        out.kv_capacity += c.kv_capacity_pages;
        // Hybrid has one exact byte gauge for KV blocks + state snapshots. Legacy capacities
        // expose Host KV alone, so never label that as total host memory or GPU utilization.
        out.host_used += s.hybrid_host_capacity_bytes != 0 ? s.hybrid_host_used_bytes
                                                         : s.host_kv_occupied_bytes;
        out.host_capacity += s.hybrid_host_capacity_bytes != 0 ? s.hybrid_host_capacity_bytes
                                                             : c.host_kv_capacity_bytes;
        out.device_state += s.device_state_occupied_slots;
        out.host_state += s.host_state_occupied_slots;
        out.spill += s.pressure_spill_pages;
        out.evict += s.pressure_private_owners_evicted + s.pressure_shared_owners_evicted;
        out.demote += s.pressure_private_owners_demoted;
        out.degrade += s.pressure_private_owners_degraded + s.pressure_shared_owners_degraded;
        out.fallback += s.pressure_maximal_fallback_selections;
        out.down += s.main_kv_d2h_bytes + s.backend_kv_d2h_bytes + s.state_d2h_bytes;
        out.up += s.main_kv_h2d_bytes + s.backend_kv_h2d_bytes + s.state_h2d_bytes;
        out.prefill_tokens += s.computed_prefill_tokens;
        out.decode_tokens += s.committed_decode_tokens;
        out.prefill_seconds += s.prefill_seconds_total;
        out.decode_seconds += s.decode_seconds_total;
    }
    return out;
}

} // namespace

std::vector<std::string> render_tui_stats_panel(const ConsoleStatsSnapshot& state,
                                               std::size_t columns, std::size_t rows,
                                               bool colours) {
    // The footer never occupies the autowrap column. Leave four console rows for ordinary logs;
    // shrink to a small summary when the terminal is too short for the full dashboard.
    const std::size_t width = std::min<std::size_t>(columns == 0 ? 79 : columns - 1, 119);
    const std::size_t height = rows == 0 ? 20 : rows > 5 ? rows - 4 : rows > 1 ? rows - 1 : 0;
    if (width < 8 || height == 0) { return {}; }
    const auto ink = [colours](std::string_view text, std::string_view style) {
        return colours ? std::string(style) + std::string(text) + std::string(kReset)
                       : std::string(text);
    };
    const Live live = live_totals(state);
    const auto& recent = state.recent.sum;
    const bool have_live = !state.models.empty();
    const bool busy = live.admitted != 0 || live.running != 0 || live.waiting != 0;
    const std::string status = have_live ? busy ? "ACTIVE" : "IDLE" : "WAITING FOR MODEL";
    const std::string outcomes = count(state.completed) + " done  " + count(state.failed) +
        " failed  " + count(state.cancelled) + " cancelled  " + count(state.rejected) + " rejected";

    if (width < 76 || height < 16) {
        std::vector<std::string> compact;
        const auto add = [&](std::string text) {
            if (compact.size() < height) { compact.push_back(product::fit_terminal_line(text, width)); }
        };
        add(ink("NInfer stats | " + status, kAccent));
        add("done " + count(state.completed) + "  failed " + count(state.failed));
        add("cancel " + count(state.cancelled) + "  rejected " + count(state.rejected));
        add("decode p50 " + quantile_rate(state, 0.5, true));
        add("prefill p50 " + quantile_rate(state, 0.5, false));
        add("TTFT p50 " + quantile_ttft(state, 0.5) + " | hit " +
            percent(recent.cache_hit_tokens, recent.prompt_tokens));
        add(have_live ? "run " + count(live.running) + "/" + count(live.concurrency) +
                           " wait " + count(live.waiting) + " ingress " + count(live.admitted)
                     : "No loaded model snapshot yet");
        add("device KV " + percent(live.kv_used, live.kv_capacity));
        add("host cache " + percent(live.host_used, live.host_capacity));
        add(ink("last " + count(state.recent_requests.size()) + " requests | read only", kMuted));
        return compact;
    }

    const std::size_t inner = width - 4;
    std::vector<std::string> lines;
    const auto rule = [&](std::string_view label) {
        std::string text = "+-- " + std::string(label) + " ";
        text = product::fit_terminal_line(text, width - 1);
        text.append(width - 1 - product::terminal_display_width(text), '-');
        lines.push_back(ink(text + "+", kAccent));
    };
    const auto row = [&](std::string_view text, std::string_view style = {}) {
        const std::string content = pad(text, inner);
        lines.push_back(ink("| ", kMuted) + (style.empty() ? content : ink(content, style)) +
                        ink(" |", kMuted));
    };

    rule("NInfer / stats");
    std::string model = state.models.size() == 1 ? state.models.front().capacity.model_id
                       : state.models.empty() ? "no loaded models"
                                              : count(state.models.size()) + " loaded models (combined)";
    model = product::format_pretty_text(model);
    row(cells({model, status + "  |  read only"}, inner));
    row(cells({"DECODE p50", "PREFILL p50", "TTFT p50", "PREFIX HIT"}, inner), kMuted);
    row(cells({quantile_rate(state, 0.5, true), quantile_rate(state, 0.5, false),
               quantile_ttft(state, 0.5), percent(recent.cache_hit_tokens, recent.prompt_tokens)}, inner),
        kValue);
    row(cells({"p95 " + quantile_rate(state, 0.95, true),
               "p95 " + quantile_rate(state, 0.95, false),
               "p95 " + quantile_ttft(state, 0.95),
               "last " + count(state.recent_requests.size()) + " requests"}, inner), kMuted);

    rule("Live engine");
    row(have_live ? "run " + count(live.running) + "/" + count(live.concurrency) +
        "  prefill " + count(live.prefill) + "  decode " + count(live.decode) +
        "  wait " + count(live.waiting) + "  ingress " + count(live.admitted)
        : "Waiting for the first loaded model snapshot", kValue);
    if (height >= 22) {
        row("Scheduler   materialize " + count(live.materializing) + "  capture " +
            count(live.capturing) + "  terminal " + count(live.terminal) +
            "  |  ingress peak sum " + count(live.peak), kMuted);
    }
    const auto bar_width = std::min<std::size_t>(18, inner / 5);
    row("Device KV   " + bar(live.kv_used, live.kv_capacity, bar_width) + "   " +
        count(live.kv_used) + "/" + count(live.kv_capacity) + " pages");
    row("Host cache  " + bar(live.host_used, live.host_capacity, bar_width) + "   " +
        gib(live.host_used) + "/" + gib(live.host_capacity) + " GiB");
    row("Pressure    spill " + count(live.spill) + "  evict " + count(live.evict) +
        "  demote " + count(live.demote) + "  degrade " + count(live.degrade) +
        "  fallback " + count(live.fallback),
        live.spill + live.evict + live.demote + live.degrade + live.fallback ? kWarning : kMuted);
    // These are cumulative engine products, not wall-clock throughput or device telemetry.
    if (height >= 20) {
        row("Engine tok/s  decode " + rate(live.decode_tokens, live.decode_seconds) +
            "  prefill " + rate(live.prefill_tokens, live.prefill_seconds), kMuted);
    }
    row("Transfers   down " + gib(live.down) + " GiB  up " + gib(live.up) +
        " GiB  |  state GPU " + count(live.device_state) + " host " + count(live.host_state), kMuted);

    const std::string drafter = state.speculative_backend == SpeculativeBackend::None
        ? "draft" : product::speculative_backend_name(state.speculative_backend);
    row(drafter + " " + percent(recent.model_accepted_tokens, recent.model_drafted_tokens) +
        "  ngram " + percent(recent.ngram_accepted_tokens, recent.ngram_drafted_tokens) +
        "  archive " + percent(recent.archive_accepted_tokens, recent.archive_drafted_tokens) +
        "  accepted (last " + count(state.recent.requests) + ")");

    rule("Recent requests");
    row(cells({"ID", "PROMPT", "OUTPUT", "TTFT", "DECODE", "CACHED"}, inner), kMuted);
    // Fixed budget: show up to three requests, reducing this before the summary or frame.
    const std::size_t available = height > lines.size() + 2 ? height - lines.size() - 2 : 0;
    const std::size_t shown = std::min({available, std::size_t{3}, state.recent_requests.size()});
    for (std::size_t index = 0; index < shown; ++index) {
        const auto& s = state.recent_requests[state.recent_requests.size() - 1 - index];
        row(cells({"#" + count(s.request_id), count(s.prompt_tokens), count(s.completion_tokens),
                   format_pretty_duration(s.ttft_seconds), rate(s.decode_tokens, s.decode_seconds),
                   percent(s.cache_hit_tokens, s.prompt_tokens)}, inner));
    }
    if (shown == 0 && available != 0) { row("No completed requests yet", kMuted); }
    row(outcomes, state.failed + state.cancelled + state.rejected ? kWarning : kMuted);
    rule("refresh 1s / recent window 10");
    return lines;
}

} // namespace ninfer::serve
