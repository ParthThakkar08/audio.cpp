#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/io/json.h"
#include "engine/models/fish_audio/loader.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

namespace json = engine::io::json;

// ============================================================================
// Helper: PCM16 Encoder (matches app/server/runtime.cpp:366-389)
// ============================================================================
std::vector<uint8_t> encode_pcm16_samples(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    std::vector<uint8_t> out;
    out.reserve(audio.samples.size() * sizeof(int16_t));
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        const auto * bytes = reinterpret_cast<const uint8_t *>(&pcm);
        out.insert(out.end(), bytes, bytes + sizeof(pcm));
    }
    return out;
}

// ============================================================================
// Helper: Simple Base64 Encoder (matches app/server/base64.cpp)
// ============================================================================
std::string base64_encode(const std::vector<uint8_t> & data) {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
        const uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(alphabet[(triple >> 18) & 0x3F]);
        out.push_back(alphabet[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < data.size()) ? alphabet[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < data.size()) ? alphabet[triple & 0x3F] : '=');
    }
    return out;
}

// ============================================================================
// Mock Fish Audio Session simulating full streaming lifecycle & context
// ============================================================================
struct MockFishAudioConversationTurn {
    std::string text;
    std::vector<std::vector<int>> codes;
};

class MockFishAudioStreamingSession final
    : public engine::runtime::IOfflineVoiceTaskSession,
      public engine::runtime::IStreamingVoiceTaskSession {
public:
    explicit MockFishAudioStreamingSession(engine::runtime::RunMode mode = engine::runtime::RunMode::Streaming)
        : mode_(mode) {}

    std::string family() const override {
        return "fish_audio";
    }

    engine::runtime::VoiceTaskKind task_kind() const override {
        return engine::runtime::VoiceTaskKind::Tts;
    }

    engine::runtime::RunMode run_mode() const override {
        return mode_;
    }

    void prepare(const engine::runtime::SessionPreparationRequest & request) override {
        (void)request;
        prepared_ = true;
    }

    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override {
        (void)request;
        if (!prepared_) {
            throw std::runtime_error("Session not prepared");
        }
        engine::runtime::TaskResult result;
        result.audio_output = engine::runtime::AudioBuffer{44100, 1, std::vector<float>(4410, 0.25f)};
        return result;
    }

    engine::runtime::StreamingPolicy streaming_policy() const override {
        engine::runtime::StreamingPolicy policy;
        policy.input = engine::runtime::StreamingInputKind::None;
        policy.output = engine::runtime::StreamingOutputKind::PullEvents;
        return policy;
    }

    void start_stream(const engine::runtime::TaskRequest & request) override {
        if (!prepared_) {
            throw std::runtime_error("Session not prepared");
        }
        if (mode_ != engine::runtime::RunMode::Streaming) {
            throw std::runtime_error("Fish Audio start_stream requires a streaming session");
        }
        reset();
        int chunk_size = 200;
        const auto it = request.options.find("text_chunk_size");
        if (it != request.options.end()) {
            try {
                chunk_size = std::stoi(it->second);
            } catch (...) {}
        }
        const auto chunk_mode = engine::text::parse_text_chunk_mode_override(request.options)
                                    .value_or(engine::text::TextChunkMode::Default);
        stream_chunk_requests_ = engine::runtime::chunk_text_request(request, chunk_size, chunk_mode);
        stream_chunk_index_ = 0;
        stream_previous_turn_ = std::nullopt;
        stream_merged_audio_ = engine::runtime::AudioBuffer{44100, 1, {}};
        stream_started_ = true;
    }

    std::optional<engine::runtime::StreamEvent> next_stream_event() override {
        if (!stream_started_) {
            throw std::runtime_error("Fish Audio streaming has not been started");
        }
        if (stream_chunk_index_ >= stream_chunk_requests_.size()) {
            return std::nullopt;
        }

        const size_t chunk_index = stream_chunk_index_++;
        const auto & chunk_req = stream_chunk_requests_[chunk_index];
        const std::string text = chunk_req.text_input.has_value() ? chunk_req.text_input->text : "";

        // Simulated audio chunk (2205 samples = 50ms at 44.1kHz)
        const size_t samples_per_chunk = 2205;
        std::vector<float> samples(samples_per_chunk, 0.1f * static_cast<float>(chunk_index + 1));
        engine::runtime::AudioBuffer chunk_audio{44100, 1, std::move(samples)};

        // Simulated acoustic codes (e.g. 10 codebooks x 100 frames)
        std::vector<std::vector<int>> codes(10, std::vector<int>(100, static_cast<int>(chunk_index)));

        engine::runtime::StreamEvent event;
        engine::runtime::NamedAudioBuffer named;
        named.id = "chunk_" + std::to_string(chunk_index);
        named.audio = chunk_audio;
        event.named_audio_outputs.push_back(std::move(named));
        // event.audio_output is intentionally nullopt (Zero-duplication contract)
        event.is_final = (stream_chunk_index_ >= stream_chunk_requests_.size());

        // Context chaining: preserve turn
        stream_previous_turn_ = MockFishAudioConversationTurn{text, std::move(codes)};
        for (float s : chunk_audio.samples) {
            stream_merged_audio_.samples.push_back(s);
        }

        return event;
    }

    void set_stream_event_sink(engine::runtime::StreamEventCallback sink) override {
        (void)sink;
    }

    engine::runtime::TaskResult finish_stream() override {
        if (!stream_started_) {
            throw std::runtime_error("Fish Audio streaming has not been started");
        }
        engine::runtime::TaskResult result;
        result.audio_output = std::move(stream_merged_audio_);
        reset();
        return result;
    }

    void reset() override {
        stream_chunk_requests_.clear();
        stream_previous_turn_ = std::nullopt;
        stream_merged_audio_ = engine::runtime::AudioBuffer{};
        stream_chunk_index_ = 0;
        stream_started_ = false;
    }

    engine::runtime::StreamEvent process_audio_chunk(const engine::runtime::AudioChunk & chunk) override {
        (void)chunk;
        return engine::runtime::StreamEvent{};
    }

    engine::runtime::TaskResult finalize() override {
        return finish_stream();
    }

    // Inspection helpers for test assertions
    bool is_started() const { return stream_started_; }
    size_t chunk_count() const { return stream_chunk_requests_.size(); }
    size_t current_chunk_index() const { return stream_chunk_index_; }
    bool has_previous_turn() const { return stream_previous_turn_.has_value(); }
    const std::optional<MockFishAudioConversationTurn> & previous_turn() const { return stream_previous_turn_; }

private:
    engine::runtime::RunMode mode_;
    bool prepared_ = false;
    std::vector<engine::runtime::TaskRequest> stream_chunk_requests_;
    std::optional<MockFishAudioConversationTurn> stream_previous_turn_;
    engine::runtime::AudioBuffer stream_merged_audio_;
    size_t stream_chunk_index_ = 0;
    bool stream_started_ = false;
};

// ============================================================================
// Test 1: Loader capabilities registration & contracts
// ============================================================================
void test_loader_capabilities() {
    std::cout << "[TEST 1] Testing Fish Audio loader capabilities and registration..." << std::endl;
    auto loader = engine::models::fish_audio::make_fish_audio_loader();
    engine::test::require(loader != nullptr, "Loader must not be null");
    engine::test::require_eq(loader->family(), std::string("fish_audio"), "Loader family must be 'fish_audio'");

    const auto capabilities = loader->advertised_capabilities();
    engine::test::require_eq(capabilities.supported_tasks.size(), static_cast<size_t>(1), "Must support 1 task entry");
    engine::test::require(capabilities.supported_tasks[0].task == engine::runtime::VoiceTaskKind::Tts, "Must support TTS task");

    const auto & modes = capabilities.supported_tasks[0].modes;
    engine::test::require_eq(modes.size(), static_cast<size_t>(2), "Must advertise exactly 2 run modes");

    const bool has_offline = std::find(modes.begin(), modes.end(), engine::runtime::RunMode::Offline) != modes.end();
    const bool has_streaming = std::find(modes.begin(), modes.end(), engine::runtime::RunMode::Streaming) != modes.end();
    engine::test::require(has_offline, "Loader must advertise RunMode::Offline");
    engine::test::require(has_streaming, "Loader must advertise RunMode::Streaming");

    engine::test::require(capabilities.supports_speaker_reference, "Loader must advertise speaker reference cloning");
    engine::test::require(capabilities.supports_style_condition, "Loader must advertise style conditioning");
    std::cout << "  -> PASSED: Loader correctly advertises offline and streaming TTS with speaker cloning." << std::endl;
}

// ============================================================================
// Test 2: Streaming session polymorphism and interface contracts
// ============================================================================
void test_session_polymorphism_and_interfaces() {
    std::cout << "[TEST 2] Testing session polymorphism and interface contracts..." << std::endl;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> base_session =
        std::make_unique<MockFishAudioStreamingSession>(engine::runtime::RunMode::Streaming);

    engine::test::require_eq(base_session->family(), std::string("fish_audio"), "Base session family");
    engine::test::require(base_session->task_kind() == engine::runtime::VoiceTaskKind::Tts, "Base session task kind");
    engine::test::require(base_session->run_mode() == engine::runtime::RunMode::Streaming, "Base session run mode");

    auto * streaming_ptr = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(base_session.get());
    engine::test::require(streaming_ptr != nullptr, "Must dynamically cast to IStreamingVoiceTaskSession");

    auto * offline_ptr = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(base_session.get());
    engine::test::require(offline_ptr != nullptr, "Must dynamically cast to IOfflineVoiceTaskSession");

    // Test streaming policy
    const auto policy = streaming_ptr->streaming_policy();
    engine::test::require(policy.input == engine::runtime::StreamingInputKind::None, "TTS streaming input must be None");
    engine::test::require(policy.output == engine::runtime::StreamingOutputKind::PullEvents, "Output must be PullEvents");

    // Test process_audio_chunk contract
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.channels = 1;
    chunk.samples = {0.1f, -0.1f, 0.0f};
    const auto event = streaming_ptr->process_audio_chunk(chunk);
    engine::test::require(event.named_audio_outputs.empty(), "process_audio_chunk returns empty event for TTS");

    std::cout << "  -> PASSED: Polymorphism and interface contracts fully verified." << std::endl;
}

// ============================================================================
// Test 3: Text chunking under default, word_budget, and tag-aware modes
// ============================================================================
void test_text_chunking_modes() {
    std::cout << "[TEST 3] Testing text chunking modes (default, word_budget, tag_aware)..." << std::endl;
    engine::runtime::TaskRequest request;

    // Case 1: Empty text
    request.text_input = engine::runtime::Transcript{""};
    auto chunks = engine::runtime::chunk_text_request(request, 200, engine::text::TextChunkMode::Default);
    engine::test::require(!chunks.empty(), "Empty text should return at least 1 request");

    // Case 2: Short text under budget
    request.text_input = engine::runtime::Transcript{"Hello world."};
    chunks = engine::runtime::chunk_text_request(request, 200, engine::text::TextChunkMode::Default);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(1), "Short text should produce 1 chunk");
    engine::test::require_eq(chunks[0].text_input->text, std::string("Hello world."), "Exact text match");

    // Case 3: Long multi-sentence paragraph with sentence/word budget splitting
    std::string long_text = "The quick brown fox jumps over the lazy dog. "
                            "Fish Audio S2 Pro provides expressive neural speech synthesis. "
                            "It supports real-time streaming output via chunked HTTP transfer. "
                            "Audio chunks are emitted incrementally during text synthesis. "
                            "Multi-turn conversational context is preserved across chunk boundaries. "
                            "This ensures natural prosody and seamless transitions between sentences.";
    request.text_input = engine::runtime::Transcript{long_text};
    chunks = engine::runtime::chunk_text_request(request, 60, engine::text::TextChunkMode::Default);
    engine::test::require(chunks.size() >= 4, "Long text should split into multiple chunks under small budget");

    std::string reconstructed;
    for (size_t i = 0; i < chunks.size(); ++i) {
        engine::test::require(chunks[i].text_input.has_value(), "Chunk must have text_input");
        engine::test::require(!chunks[i].text_input->text.empty(), "Chunk text must not be empty");
        if (!reconstructed.empty()) reconstructed += " ";
        reconstructed += chunks[i].text_input->text;
    }
    engine::test::require(reconstructed.find("Fish Audio S2 Pro") != std::string::npos, "Content preservation");

    // Case 4: Tag-aware chunking mode with XML voice tags
    request.options["text_chunk_mode"] = "tag_aware";
    request.text_input = engine::runtime::Transcript{
        "<voice name=\"prakash\">First paragraph with speaker tag.</voice>"
        "<voice name=\"narrator\">Second paragraph with different speaker tag.</voice>"
    };
    chunks = engine::runtime::chunk_text_request(request, 40, engine::text::TextChunkMode::TagAware);
    engine::test::require(chunks.size() >= 2, "Tag-aware chunking should segment across voice tags");

    std::cout << "  -> PASSED: All text chunking modes verified across boundary and edge conditions." << std::endl;
}

// ============================================================================
// Test 4: Mock callback and event emission verification (Zero-duplication)
// ============================================================================
void test_mock_event_emission_and_callback() {
    std::cout << "[TEST 4] Testing mock callback and event emission verification..." << std::endl;
    MockFishAudioStreamingSession session;
    session.prepare({});

    engine::runtime::TaskRequest request;
    request.options["text_chunk_size"] = "50";
    request.text_input = engine::runtime::Transcript{
        "First sentence to stream. Second sentence to follow. Third sentence to conclude."
    };

    session.start_stream(request);
    engine::test::require(session.is_started(), "Session must be started");
    const size_t expected_chunks = session.chunk_count();
    engine::test::require(expected_chunks >= 2, "Must produce at least 2 chunks");

    int sink_callbacks = 0;
    std::vector<engine::runtime::StreamEvent> events;
    auto sink = [&](const engine::runtime::StreamEvent & ev) {
        sink_callbacks++;
        events.push_back(ev);
    };
    session.set_stream_event_sink(sink);

    size_t chunk_counter = 0;
    while (auto ev = session.next_stream_event()) {
        sink(*ev);
        // 1. Verify audio_output is nullopt
        engine::test::require(!ev->audio_output.has_value(), "audio_output MUST be nullopt to prevent duplicate emission");
        // 2. Verify named_audio_outputs has exactly 1 buffer
        engine::test::require_eq(ev->named_audio_outputs.size(), static_cast<size_t>(1), "named_audio_outputs size == 1");
        engine::test::require_eq(ev->named_audio_outputs[0].id, "chunk_" + std::to_string(chunk_counter), "Chunk ID ordering");
        engine::test::require_eq(ev->named_audio_outputs[0].audio.samples.size(), static_cast<size_t>(2205), "Sample count per chunk");

        // 3. Verify is_final flag
        const bool should_be_final = (chunk_counter + 1 == expected_chunks);
        engine::test::require_eq(ev->is_final, should_be_final, "is_final flag correctness");
        chunk_counter++;
    }

    engine::test::require_eq(chunk_counter, expected_chunks, "All chunks consumed");
    engine::test::require_eq(sink_callbacks, static_cast<int>(expected_chunks), "Exact 1:1 sink callback ratio");

    // 4. Verify stream termination returns nullopt
    auto after_final = session.next_stream_event();
    engine::test::require(!after_final.has_value(), "next_stream_event after completion must return nullopt");

    // 5. Verify finish_stream produces full concatenated audio
    auto result = session.finish_stream();
    engine::test::require(result.audio_output.has_value(), "finish_stream must produce audio_output");
    engine::test::require_eq(result.audio_output->samples.size(), expected_chunks * 2205, "Concatenated sample count");
    engine::test::require(!session.is_started(), "finish_stream must reset started flag");

    std::cout << "  -> PASSED: Exactly 1 NamedAudioBuffer emitted per chunk, is_final accurate, zero duplication." << std::endl;
}

// ============================================================================
// Test 5: Multi-turn conversation context chaining across chunks
// ============================================================================
void test_multi_turn_context_chaining() {
    std::cout << "[TEST 5] Testing multi-turn conversation context chaining across chunks..." << std::endl;
    MockFishAudioStreamingSession session;
    session.prepare({});

    engine::runtime::TaskRequest request;
    request.options["text_chunk_size"] = "40";
    request.text_input = engine::runtime::Transcript{
        "Turn one sentence. Turn two sentence. Turn three sentence."
    };

    session.start_stream(request);
    engine::test::require(!session.has_previous_turn(), "Initial turn must have no previous turn");

    // Chunk 0
    auto ev0 = session.next_stream_event();
    engine::test::require(ev0.has_value(), "Chunk 0 event");
    engine::test::require(session.has_previous_turn(), "Context must be set after chunk 0");
    const auto & turn0 = *session.previous_turn();
    engine::test::require(!turn0.text.empty(), "Turn 0 text stored");
    engine::test::require_eq(turn0.codes.size(), static_cast<size_t>(10), "Turn 0 acoustic codes stored (10 codebooks)");

    // Chunk 1
    auto ev1 = session.next_stream_event();
    engine::test::require(ev1.has_value(), "Chunk 1 event");
    engine::test::require(session.has_previous_turn(), "Context must be updated after chunk 1");
    const auto & turn1 = *session.previous_turn();
    engine::test::require_eq(turn1.codes[0][0], 1, "Turn 1 codes reflect chunk 1 index");

    // Finish stream
    (void)session.finish_stream();
    engine::test::require(!session.has_previous_turn(), "finish_stream must reset conversation context");

    std::cout << "  -> PASSED: Multi-turn acoustic context chaining maintained and reset correctly." << std::endl;
}

// ============================================================================
// Test 6: Session lifecycle error handling and reset safety
// ============================================================================
void test_session_lifecycle_and_error_handling() {
    std::cout << "[TEST 6] Testing session lifecycle error handling and reset safety..." << std::endl;
    MockFishAudioStreamingSession session(engine::runtime::RunMode::Streaming);

    // 1. Calling next_stream_event() without start_stream() throws
    bool caught_unstarted_next = false;
    try {
        session.next_stream_event();
    } catch (const std::runtime_error & ex) {
        caught_unstarted_next = (std::string(ex.what()).find("Fish Audio streaming has not been started") != std::string::npos);
    }
    engine::test::require(caught_unstarted_next, "next_stream_event before start must throw");

    // 2. Calling finish_stream() without start_stream() throws
    bool caught_unstarted_finish = false;
    try {
        session.finish_stream();
    } catch (const std::runtime_error & ex) {
        caught_unstarted_finish = (std::string(ex.what()).find("Fish Audio streaming has not been started") != std::string::npos);
    }
    engine::test::require(caught_unstarted_finish, "finish_stream before start must throw");

    // 3. Double reset safety
    session.reset();
    session.reset();
    engine::test::require(!session.is_started(), "Double reset must be safe and idempotent");

    // 4. Calling start_stream on offline session throws
    MockFishAudioStreamingSession offline_session(engine::runtime::RunMode::Offline);
    offline_session.prepare({});
    bool caught_offline_start = false;
    try {
        engine::runtime::TaskRequest req;
        req.text_input = engine::runtime::Transcript{"test"};
        offline_session.start_stream(req);
    } catch (const std::runtime_error & ex) {
        caught_offline_start = (std::string(ex.what()).find("requires a streaming session") != std::string::npos);
    }
    engine::test::require(caught_offline_start, "start_stream on offline session must throw");

    // 5. Finalize delegation
    session.prepare({});
    engine::runtime::TaskRequest valid_req;
    valid_req.text_input = engine::runtime::Transcript{"Hello from finalize test."};
    session.start_stream(valid_req);
    while (session.next_stream_event()) {}
    auto fin_res = session.finalize();
    engine::test::require(fin_res.audio_output.has_value(), "finalize() must return audio_output");
    engine::test::require(!session.is_started(), "finalize() must leave session in unstarted state");

    std::cout << "  -> PASSED: Lifecycle guards, double reset, and error handling fully verified." << std::endl;
}

// ============================================================================
// Test 7: Model specification JSON catalog verification
// ============================================================================
void test_model_spec_json_verification() {
    std::cout << "[TEST 7] Testing model specification JSON catalog verification..." << std::endl;
    const std::filesystem::path root(ENGINE_REPO_ROOT);
    const std::vector<std::filesystem::path> spec_paths = {
        root / "model_specs" / "fish_audio.json",
        root / "model_specs_v1" / "fish_audio.json"
    };

    for (const auto & path : spec_paths) {
        std::cout << "  Checking spec: " << path.string() << std::endl;
        engine::test::require(std::filesystem::exists(path), "Spec file must exist: " + path.string());

        const auto val = json::parse_file(path);
        engine::test::require(val.is_object(), "Spec must be a JSON object");

        // Verify family & category
        engine::test::require_eq(json::require_string(val, "family"), std::string("fish_audio"), "family must be 'fish_audio'");
        engine::test::require_eq(json::require_string(val, "category"), std::string("tts"), "category must be 'tts'");

        // Verify modes includes "offline" and "streaming"
        const auto & modes_val = val.require("modes");
        engine::test::require(modes_val.is_array(), "modes must be an array");
        bool has_offline = false;
        bool has_streaming = false;
        for (const auto & m : modes_val.as_array()) {
            if (m.is_string() && m.as_string() == "offline") has_offline = true;
            if (m.is_string() && m.as_string() == "streaming") has_streaming = true;
        }
        engine::test::require(has_offline, "modes must contain 'offline'");
        engine::test::require(has_streaming, "modes must contain 'streaming'");

        // Verify runtime tags include "gguf" and "stream"
        const auto & runtime_val = val.require("runtime");
        engine::test::require(runtime_val.is_object(), "runtime must be an object");
        const auto & tags_val = runtime_val.require("tags");
        engine::test::require(tags_val.is_array(), "runtime.tags must be an array");
        bool has_gguf = false;
        bool has_stream = false;
        for (const auto & t : tags_val.as_array()) {
            if (t.is_string() && t.as_string() == "gguf") has_gguf = true;
            if (t.is_string() && t.as_string() == "stream") has_stream = true;
        }
        engine::test::require(has_gguf, "runtime.tags must contain 'gguf'");
        engine::test::require(has_stream, "runtime.tags must contain 'stream'");

        // Verify cloning capabilities
        const auto * cap_val = val.find("capabilities");
        if (cap_val != nullptr && cap_val->is_object()) {
            const auto * clone_val = cap_val->find("clone");
            if (clone_val != nullptr && clone_val->is_array()) {
                bool has_spk_ref = false;
                for (const auto & item : clone_val->as_array()) {
                    if (item.is_string() && item.as_string() == "speaker_reference") has_spk_ref = true;
                }
                engine::test::require(has_spk_ref, "capabilities.clone must contain 'speaker_reference'");
            }
        }
    }

    std::cout << "  -> PASSED: model_specs/fish_audio.json and model_specs_v1/fish_audio.json validated." << std::endl;
}

// ============================================================================
// Test 8: Server chunked streaming response simulation (PCM16 and SSE)
// ============================================================================
void test_server_streaming_response_simulation() {
    std::cout << "[TEST 8] Testing server chunked streaming response simulation..." << std::endl;
    MockFishAudioStreamingSession session;
    session.prepare({});

    engine::runtime::TaskRequest request;
    request.options["text_chunk_size"] = "30";
    request.text_input = engine::runtime::Transcript{
        "First test sentence for streaming simulation. "
        "Second test sentence for streaming simulation. "
        "Third test sentence for streaming simulation. "
        "Fourth test sentence for streaming simulation."
    };

    session.start_stream(request);
    const size_t num_chunks = session.chunk_count();
    engine::test::require(num_chunks >= 3, "Should produce at least 3 chunks");

    // 1. Simulate Server SSE Serialization
    std::vector<std::string> sse_messages;
    size_t sse_audio_bytes = 0;
    while (auto ev = session.next_stream_event()) {
        std::vector<engine::runtime::AudioBuffer> buffers;
        if (ev->audio_output.has_value()) {
            buffers.push_back(*ev->audio_output);
        }
        for (const auto & named : ev->named_audio_outputs) {
            buffers.push_back(named.audio);
        }

        for (const auto & audio : buffers) {
            const auto pcm = encode_pcm16_samples(audio);
            sse_audio_bytes += pcm.size();
            const std::string b64 = base64_encode(pcm);
            const std::string sse_delta = "data: {\"type\":\"speech.audio.delta\",\"audio\":\"" + b64 + "\"}\n\n";
            sse_messages.push_back(sse_delta);
        }
    }
    sse_messages.push_back("data: {\"type\":\"speech.audio.done\",\"timing\":{\"ttft_ms\":85.5}}\n\n");
    sse_messages.push_back("data: [DONE]\n\n");

    engine::test::require_eq(sse_messages.size(), num_chunks + 2, "SSE message count == num_chunks + done + [DONE]");
    engine::test::require_eq(sse_audio_bytes, num_chunks * 2205 * 2, "Total SSE PCM bytes == num_chunks * 2205 samples * 2 bytes");

    (void)session.finish_stream();

    // 2. Simulate Server Raw Chunked PCM Serialization
    session.start_stream(request);
    std::vector<std::vector<uint8_t>> raw_http_chunks;
    size_t total_raw_pcm_bytes = 0;
    while (auto ev = session.next_stream_event()) {
        if (ev->audio_output.has_value()) {
            const auto pcm = encode_pcm16_samples(*ev->audio_output);
            raw_http_chunks.push_back(pcm);
            total_raw_pcm_bytes += pcm.size();
        }
        for (const auto & named : ev->named_audio_outputs) {
            const auto pcm = encode_pcm16_samples(named.audio);
            raw_http_chunks.push_back(pcm);
            total_raw_pcm_bytes += pcm.size();
        }
    }
    (void)session.finish_stream();

    engine::test::require_eq(raw_http_chunks.size(), num_chunks, "Raw HTTP chunk count == num_chunks");
    engine::test::require_eq(total_raw_pcm_bytes, num_chunks * 2205 * 2, "Total raw PCM bytes");

    std::cout << "  -> PASSED: Server SSE and raw PCM16 chunked streaming serialization verified (1.0x ratio)." << std::endl;
}

}  // namespace

int main() {
    try {
        std::cout << "==========================================================" << std::endl;
        std::cout << " Fish Audio Streaming TTS Automated Unit & Mock Test Suite" << std::endl;
        std::cout << "==========================================================" << std::endl;

        test_loader_capabilities();
        test_session_polymorphism_and_interfaces();
        test_text_chunking_modes();
        test_mock_event_emission_and_callback();
        test_multi_turn_context_chaining();
        test_session_lifecycle_and_error_handling();
        test_model_spec_json_verification();
        test_server_streaming_response_simulation();

        std::cout << "==========================================================" << std::endl;
        std::cout << " ALL 8 TEST SUITES PASSED (100% SUCCESS)" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "FAIL: " << ex.what() << std::endl;
        return 1;
    }
}
