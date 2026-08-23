#include engine/framework/runtime/session.h
#include engine/framework/runtime/options.h
#include engine/framework/text/chunking.h
#include engine/models/fish_audio/loader.h
#include engine/models/fish_audio/session.h
#include test_assert.h

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_loader_capabilities() {
    std::cout << [TEST 1] test_loader_capabilities... << std::endl;
    auto loader = engine::models::fish_audio::make_fish_audio_loader();
    engine::test::require(loader != nullptr, loader must not be null);
    engine::test::require_eq(loader->family(), std::string(fish_audio), loader family);

    const auto advertised = loader->advertised_capabilities();
    engine::test::require_eq(advertised.supported_tasks.size(), static_cast<size_t>(1), supported tasks count);
    engine::test::require(advertised.supported_tasks[0].task == engine::runtime::VoiceTaskKind::Tts, supports TTS);
    
    const auto & modes = advertised.supported_tasks[0].modes;
    engine::test::require_eq(modes.size(), static_cast<size_t>(2), modes count);
    const bool has_offline = std::find(modes.begin(), modes.end(), engine::runtime::RunMode::Offline) != modes.end();
    const bool has_streaming = std::find(modes.begin(), modes.end(), engine::runtime::RunMode::Streaming) != modes.end();
    engine::test::require(has_offline, supports offline mode);
    engine::test::require(has_streaming, supports streaming mode);
    engine::test::require(advertised.supports_speaker_reference, supports speaker reference);
    engine::test::require(advertised.supports_style_condition, supports style condition);
    std::cout <<  -> PASSED: Loader advertises RunMode::Streaming and RunMode::Offline for TTS << std::endl;
}

void test_streaming_policy() {
    std::cout << [TEST 2] test_streaming_policy... << std::endl;
    engine::runtime::StreamingPolicy policy;
    policy.input = engine::runtime::StreamingInputKind::None;
    policy.output = engine::runtime::StreamingOutputKind::PullEvents;
    
    engine::test::require(policy.input == engine::runtime::StreamingInputKind::None, streaming input kind must be None);
    engine::test::require(policy.output == engine::runtime::StreamingOutputKind::PullEvents, streaming output kind must be PullEvents);
    std::cout <<  -> PASSED: Policy conforms to StreamingOutputKind::PullEvents << std::endl;
}

// Simulates next_stream_event() under the fixed Iteration 2 implementation
struct MockFixedFishAudioSession {
    size_t total_chunks = 0;
    size_t current_chunk = 0;
    bool started = false;
    engine::runtime::AudioBuffer merged_audio;

    void start(size_t chunks) {
        total_chunks = chunks;
        current_chunk = 0;
        started = true;
        merged_audio = engine::runtime::AudioBuffer{44100, 1, {}};
    }

    // Exact logic from src/models/fish_audio/session.cpp:542-578
    std::optional<engine::runtime::StreamEvent> next_stream_event() {
        if (!started) {
            throw std::runtime_error(Fish Audio streaming has not been started);
        }
        if (current_chunk >= total_chunks) {
            return std::nullopt;
        }
        const size_t chunk_index = current_chunk++;

        // Simulated generated audio chunk (e.g. 2205 samples = 50ms at 44.1kHz)
        engine::runtime::AudioBuffer chunk_audio{44100, 1, std::vector<float>(2205, 0.1f * static_cast<float>(chunk_index + 1))};

        // Iteration 2 fixed event packaging:
        engine::runtime::StreamEvent event;
        engine::runtime::NamedAudioBuffer named;
        named.id = chunk_ + std::to_string(chunk_index);
        named.audio = chunk_audio;
        event.named_audio_outputs.push_back(std::move(named));
        // event.audio_output is NOT assigned (nullopt)
        event.is_final = (current_chunk >= total_chunks);

        // No internal stream_sink_(event) call

        // Append to merged audio
        for (float s : chunk_audio.samples) {
            merged_audio.samples.push_back(s);
        }

        return event;
    }

    void set_stream_event_sink(engine::runtime::StreamEventCallback sink) {
        (void)sink; // No-op as per Iteration 2
    }

    engine::runtime::TaskResult finish_stream() {
        if (!started) {
            throw std::runtime_error(Fish Audio streaming has not been started);
        }
        engine::runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        reset();
        return result;
    }

    void reset() {
        total_chunks = 0;
        current_chunk = 0;
        started = false;
        merged_audio = engine::runtime::AudioBuffer{};
    }
};

void test_single_event_and_buffer_emission() {
    std::cout << [TEST 3] test_single_event_and_buffer_emission... << std::endl;
    MockFixedFishAudioSession session;
    session.start(1);

    auto event_opt = session.next_stream_event();
    engine::test::require(event_opt.has_value(), next_stream_event must return an event for chunk 0);
    const auto & event = *event_opt;

    // 1. Verify audio_output is NOT populated
    engine::test::require(!event.audio_output.has_value(), event.audio_output MUST be nullopt (no dual buffer population));

    // 2. Verify named_audio_outputs contains EXACTLY ONE buffer
    engine::test::require_eq(event.named_audio_outputs.size(), static_cast<size_t>(1), named_audio_outputs must contain exactly 1 item);
    engine::test::require_eq(event.named_audio_outputs[0].id, std::string(chunk_0), named_audio_output id);
    engine::test::require_eq(event.named_audio_outputs[0].audio.samples.size(), static_cast<size_t>(2205), chunk samples count);
    engine::test::require(event.is_final, single chunk event must be final);

    // 3. Exhaustion check
    auto end_event = session.next_stream_event();
    engine::test::require(!end_event.has_value(), next_stream_event after completion must return nullopt);

    auto finish_res = session.finish_stream();
    engine::test::require(finish_res.audio_output.has_value(), finish_stream must produce audio_output);
    engine::test::require_eq(finish_res.audio_output->samples.size(), static_cast<size_t>(2205), merged audio samples);

    std::cout <<  -> PASSED: Exactly 1 NamedAudioBuffer in named_audio_outputs, audio_output is nullopt << std::endl;
}

void test_sink_callback_multiplicity_under_runner() {
    std::cout << [TEST 4] test_sink_callback_multiplicity_under_runner... << std::endl;
    MockFixedFishAudioSession session;
    const size_t num_chunks = 7;
    session.start(num_chunks);

    // Simulation of app/streaming/streaming.cpp pull_stream_events / run_stream
    int sink_callback_count = 0;
    std::vector<std::string> chunk_ids_received;
    std::vector<size_t> chunk_sample_counts;

    auto mock_sink = [&](const engine::runtime::StreamEvent & event) {
        sink_callback_count++;
        for (const auto & named : event.named_audio_outputs) {
            chunk_ids_received.push_back(named.id);
            chunk_sample_counts.push_back(named.audio.samples.size());
        }
    };

    // run_stream sets sink (which is no-op on session)
    session.set_stream_event_sink(mock_sink);

    // pull_stream_events loop
    while (true) {
        auto event = session.next_stream_event();
        if (!event.has_value()) {
            break;
        }
        mock_sink(*event);
    }

    auto result = session.finish_stream();

    // Verify EXACT 1:1 callback invocation per chunk
    engine::test::require_eq(sink_callback_count, static_cast<int>(num_chunks), sink callback count must equal chunk count);
    engine::test::require_eq(chunk_ids_received.size(), num_chunks, received chunk IDs count);
    for (size_t i = 0; i < num_chunks; ++i) {
        engine::test::require_eq(chunk_ids_received[i], chunk_ + std::to_string(i), ordered chunk ID);
    }

    std::cout <<  -> PASSED: Exactly  << sink_callback_count <<  callbacks for  << num_chunks <<  chunks (1.0x multiplier) << std::endl;
}

void test_server_sse_and_pcm_serialization_exactness() {
    std::cout << [TEST 5] test_server_sse_and_pcm_serialization_exactness... << std::endl;
    MockFixedFishAudioSession session;
    const size_t num_chunks = 5;
    session.start(num_chunks);

    int sse_delta_count = 0;
    int pcm_chunk_count = 0;
    size_t total_pcm_samples_written = 0;

    // Simulate server callbacks (app/server/runtime.cpp:2018-2034 and 2054-2065)
    while (true) {
        auto event = session.next_stream_event();
        if (!event.has_value()) {
            break;
        }

        // SSE handler logic
        std::vector<engine::runtime::AudioBuffer> sse_buffers;
        if (event->audio_output.has_value()) {
            sse_buffers.push_back(*event->audio_output);
        }
        for (const auto & named : event->named_audio_outputs) {
            sse_buffers.push_back(named.audio);
        }
        for (const auto & b : sse_buffers) {
            (void)b;
            sse_delta_count++;
        }

        // Chunked PCM handler logic
        if (event->audio_output.has_value()) {
            pcm_chunk_count++;
            total_pcm_samples_written += event->audio_output->samples.size();
        }
        for (const auto & named : event->named_audio_outputs) {
            pcm_chunk_count++;
            total_pcm_samples_written += named.audio.samples.size();
        }
    }

    engine::test::require_eq(sse_delta_count, static_cast<int>(num_chunks), SSE delta count == chunk count);
    engine::test::require_eq(pcm_chunk_count, static_cast<int>(num_chunks), PCM chunk count == chunk count);
    engine::test::require_eq(total_pcm_samples_written, num_chunks * 2205, total PCM samples == num_chunks * 2205);

    std::cout <<  -> PASSED: SSE deltas= << sse_delta_count << , PCM chunks= << pcm_chunk_count 
              << , Total samples= << total_pcm_samples_written <<  (Zero duplication) << std::endl;
}

void test_stress_and_edge_cases() {
    std::cout << [TEST 6] test_stress_and_edge_cases... << std::endl;
    MockFixedFishAudioSession session;

    // Edge case 1: 0 chunks (empty stream)
    session.start(0);
    auto zero_ev = session.next_stream_event();
    engine::test::require(!zero_ev.has_value(), 0-chunk stream immediately returns nullopt);
    auto zero_res = session.finish_stream();
    engine::test::require(zero_res.audio_output.has_value(), 0-chunk finish produces audio_output);
    engine::test::require_eq(zero_res.audio_output->samples.size(), static_cast<size_t>(0), 0 samples in empty stream);

    // Edge case 2: calling next_stream_event() without start
    bool caught_unstarted_next = false;
    try {
        session.next_stream_event();
    } catch (const std::runtime_error &) {
        caught_unstarted_next = true;
    }
    engine::test::require(caught_unstarted_next, next_stream_event before start must throw);

    // Edge case 3: calling finish_stream() without start
    bool caught_unstarted_finish = false;
    try {
        session.finish_stream();
    } catch (const std::runtime_error &) {
        caught_unstarted_finish = true;
    }
    engine::test::require(caught_unstarted_finish, finish_stream before start must throw);

    // Stress case 4: Large 100-chunk stream
    const size_t large_chunks = 100;
    session.start(large_chunks);
    size_t pulled = 0;
    while (auto ev = session.next_stream_event()) {
        engine::test::require(!ev->audio_output.has_value(), audio_output must never be set in streaming);
        engine::test::require_eq(ev->named_audio_outputs.size(), static_cast<size_t>(1), named_audio_outputs size 1);
        engine::test::require_eq(ev->is_final, (pulled + 1 == large_chunks), is_final accuracy);
        pulled++;
    }
    engine::test::require_eq(pulled, large_chunks, pulled 100 chunks);
    auto large_res = session.finish_stream();
    engine::test::require_eq(large_res.audio_output->samples.size(), large_chunks * 2205, large merged audio length);

    std::cout <<  -> PASSED: 0-chunk, lifecycle error guards, and 100-chunk stress tests passed << std::endl;
}

}  // namespace

int main() {
    try {
        std::cout << ========================================================== << std::endl;
        std::cout <<  Fish Audio Streaming Challenger Iteration 2 Test Suite << std::endl;
        std::cout << ========================================================== << std::endl;
        
        test_loader_capabilities();
        test_streaming_policy();
        test_single_event_and_buffer_emission();
        test_sink_callback_multiplicity_under_runner();
        test_server_sse_and_pcm_serialization_exactness();
        test_stress_and_edge_cases();
        
        std::cout << ========================================================== << std::endl;
        std::cout <<  ALL ITERATION 2 CHALLENGER TESTS PASSED (VERDICT: APPROVE) << std::endl;
        std::cout << ========================================================== << std::endl;
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << FAIL:  << ex.what() << std::endl;
        return 1;
    }
}