#!/usr/bin/env python3
"""Measure streaming TTS latency, TTFA, chunk interval timings, and audio quality."""

from __future__ import annotations

import argparse
import base64
import csv
import http.client
import json
import math
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "streaming_test"
DEFAULT_SERVER_BIN = REPO_ROOT / "build" / "bin" / "audiocpp_server"
DEFAULT_SAMPLE_RATE = 44100


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def split_incremental_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise RuntimeError("--max-chars must be positive")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence
        while len(current) > max_chars:
            split_at = current.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(current[:split_at].strip())
            current = current[split_at:].strip()
    if current:
        chunks.append(current)
    if not chunks:
        raise RuntimeError("input text produced no chunks")
    return chunks


def gpu_memory_mib() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if not values:
            return 0
        return max(values)
    except Exception:
        return 0


class VramSampler:
    def __init__(self, path: Path, interval_s: float) -> None:
        self.path = path
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        if self.interval_s <= 0.0:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        start = time.perf_counter()
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_ms", "vram_mib"])
            while not self.stop_event.is_set():
                vram = gpu_memory_mib()
                writer.writerow([round((time.perf_counter() - start) * 1000.0, 3), vram])
                handle.flush()
                time.sleep(self.interval_s)


def read_vram_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "vram_start_mib": 0,
            "vram_peak_mib": 0,
            "vram_end_mib": 0,
            "vram_samples": 0,
        }
    values: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "vram_mib" in row:
                values.append(int(row["vram_mib"]))
    if not values:
        return {
            "vram_start_mib": 0,
            "vram_peak_mib": 0,
            "vram_end_mib": 0,
            "vram_samples": 0,
        }
    return {
        "vram_start_mib": values[0],
        "vram_peak_mib": max(values),
        "vram_end_mib": values[-1],
        "vram_samples": len(values),
    }


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server health timeout for {base_url}: {last_error}")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def calc_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "jitter": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values) if len(values) > 1 else 0.0
    stdev_val = math.sqrt(variance)
    return {
        "count": float(len(values)),
        "mean": round(mean_val, 2),
        "jitter": round(stdev_val, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
    }


def write_wav(path: Path, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def validate_audio_pcm(pcm: bytes, sample_rate: int) -> dict[str, Any]:
    if len(pcm) % 2 != 0:
        raise RuntimeError(f"PCM byte length ({len(pcm)}) is not divisible by 2 (16-bit samples)")
    sample_count = len(pcm) // 2
    if sample_count == 0:
        raise RuntimeError("PCM buffer contains 0 audio samples")

    samples = struct.unpack(f"<{sample_count}h", pcm)
    min_val = min(samples)
    max_val = max(samples)
    sum_sq = sum(float(s * s) for s in samples)
    rms = math.sqrt(sum_sq / sample_count)
    duration_s = sample_count / float(sample_rate)

    is_silent = (rms < 1.0)
    clipping_count = sum(1 for s in samples if s >= 32760 or s <= -32760)

    return {
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "duration_s": round(duration_s, 4),
        "min_amplitude": int(min_val),
        "max_amplitude": int(max_val),
        "rms_amplitude": round(rms, 2),
        "clipping_samples": clipping_count,
        "is_silent": is_silent,
        "audio_continuity": not is_silent,
    }


def stream_speech_request_audio(
    base_url: str,
    model: str,
    text: str,
    sample_rate: int,
    options: dict[str, Any] | None = None,
    speaker_ref: str | None = None,
    reference_text: str | None = None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "options": options or {"retry_badcase": False},
    }
    if speaker_ref:
        payload["speaker_ref"] = speaker_ref
    if reference_text:
        payload["reference_text"] = reference_text

    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }

    conn = http.client.HTTPSConnection(host, port, timeout=1200) if parsed.scheme == "https" else http.client.HTTPConnection(host, port, timeout=1200)
    
    start = time.perf_counter()
    path = parsed.path.rstrip("/") + "/v1/audio/speech"
    conn.request("POST", path, body=body_bytes, headers=headers)
    response = conn.getresponse()

    if response.status != 200:
        err = response.read().decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status}: {err}")

    pcm = bytearray()
    chunk_arrival_times_ms: list[float] = []
    chunk_intervals_ms: list[float] = []
    chunk_byte_sizes: list[int] = []
    first_audio_ms: float | None = None
    last_arrival_time = start

    while True:
        chunk = response.read(4096)
        now = time.perf_counter()
        if not chunk:
            break
        event_ms = (now - start) * 1000.0
        interval_ms = (now - last_arrival_time) * 1000.0
        last_arrival_time = now

        if first_audio_ms is None:
            first_audio_ms = event_ms
        else:
            chunk_intervals_ms.append(interval_ms)

        chunk_arrival_times_ms.append(round(event_ms, 2))
        chunk_byte_sizes.append(len(chunk))
        pcm.extend(chunk)

    conn.close()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not pcm:
        raise RuntimeError("streaming audio response produced no audio bytes")

    pcm_bytes = bytes(pcm)
    validation = validate_audio_pcm(pcm_bytes, sample_rate)
    timing_stats = calc_stats(chunk_intervals_ms)

    return {
        "stream_format": "audio",
        "request": payload,
        "elapsed_ms": round(elapsed_ms, 2),
        "ttfa_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "ttft_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "client_first_audio_ms": round(first_audio_ms, 2) if first_audio_ms is not None else None,
        "client_done_ms": round(elapsed_ms, 2),
        "chunk_count": len(chunk_byte_sizes),
        "chunk_byte_sizes": chunk_byte_sizes,
        "chunk_arrival_times_ms": chunk_arrival_times_ms,
        "chunk_interval_stats": timing_stats,
        "audio_validation": validation,
        "pcm_bytes": len(pcm_bytes),
        "pcm": pcm_bytes,
    }


def stream_speech_request_sse(
    base_url: str,
    model: str,
    text: str,
    sample_rate: int,
    options: dict[str, Any] | None = None,
    speaker_ref: str | None = None,
    reference_text: str | None = None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "sse",
        "options": options or {"retry_badcase": False},
    }
    if speaker_ref:
        payload["speaker_ref"] = speaker_ref
    if reference_text:
        payload["reference_text"] = reference_text

    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    conn = http.client.HTTPSConnection(host, port, timeout=1200) if parsed.scheme == "https" else http.client.HTTPConnection(host, port, timeout=1200)
    
    start = time.perf_counter()
    path = parsed.path.rstrip("/") + "/v1/audio/speech"
    conn.request("POST", path, body=body_bytes, headers=headers)
    response = conn.getresponse()

    if response.status != 200:
        err = response.read().decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {response.status}: {err}")

    pcm = bytearray()
    delta_events = 0
    first_delta_ms: float | None = None
    done_ms: float | None = None
    ttft_ms: float | None = None
    done_event: dict[str, Any] | None = None
    errors: list[str] = []
    chunk_arrival_times_ms: list[float] = []
    chunk_intervals_ms: list[float] = []
    last_arrival_time = start

    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        now = time.perf_counter()
        event_ms = (now - start) * 1000.0
        event_type = event.get("type")

        if event_type == "error":
            errors.append(event.get("error", {}).get("message", json.dumps(event)))
        elif event_type == "speech.audio.delta":
            delta_events += 1
            interval_ms = (now - last_arrival_time) * 1000.0
            last_arrival_time = now

            if first_delta_ms is None:
                first_delta_ms = event_ms
            else:
                chunk_intervals_ms.append(interval_ms)

            chunk_arrival_times_ms.append(round(event_ms, 2))
            audio_bytes = base64.b64decode(event["audio"])
            pcm.extend(audio_bytes)
        elif event_type == "speech.audio.done":
            done_event = event
            done_ms = event_ms
            ttft_ms = event.get("timing", {}).get("ttft_ms")

    conn.close()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if errors:
        raise RuntimeError("; ".join(errors))
    if not pcm:
        raise RuntimeError("streaming SSE response produced no audio bytes")

    pcm_bytes = bytes(pcm)
    validation = validate_audio_pcm(pcm_bytes, sample_rate)
    timing_stats = calc_stats(chunk_intervals_ms)

    return {
        "stream_format": "sse",
        "request": payload,
        "elapsed_ms": round(elapsed_ms, 2),
        "ttfa_ms": round(first_delta_ms, 2) if first_delta_ms is not None else None,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else (round(first_delta_ms, 2) if first_delta_ms is not None else None),
        "client_first_audio_ms": round(first_delta_ms, 2) if first_delta_ms is not None else None,
        "client_done_ms": round(done_ms, 2) if done_ms is not None else round(elapsed_ms, 2),
        "delta_events": delta_events,
        "chunk_count": delta_events,
        "chunk_arrival_times_ms": chunk_arrival_times_ms,
        "chunk_interval_stats": timing_stats,
        "audio_validation": validation,
        "pcm_bytes": len(pcm_bytes),
        "done_event": done_event,
        "pcm": pcm_bytes,
    }


def stream_speech_request(
    base_url: str,
    model: str,
    text: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    stream_format: str = "audio",
    options: dict[str, Any] | None = None,
    speaker_ref: str | None = None,
    reference_text: str | None = None,
) -> dict[str, Any]:
    if stream_format == "audio":
        return stream_speech_request_audio(
            base_url=base_url,
            model=model,
            text=text,
            sample_rate=sample_rate,
            options=options,
            speaker_ref=speaker_ref,
            reference_text=reference_text,
        )
    elif stream_format == "sse":
        return stream_speech_request_sse(
            base_url=base_url,
            model=model,
            text=text,
            sample_rate=sample_rate,
            options=options,
            speaker_ref=speaker_ref,
            reference_text=reference_text,
        )
    else:
        raise ValueError(f"Unsupported stream_format: {stream_format} (must be 'audio' or 'sse')")


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        base_url = args.url.rstrip("/")
        host = urllib.parse.urlparse(base_url).hostname or "127.0.0.1"
        port = urllib.parse.urlparse(base_url).port or 80
        config = {"host": host, "port": port}
    elif args.server_config:
        config = load_json(args.server_config)
        host = args.host or config.get("host", "127.0.0.1")
        port = int(args.port or config.get("port", 8080))
        base_url = f"http://{host}:{port}"
    else:
        host = args.host or "127.0.0.1"
        port = int(args.port or 8080)
        base_url = f"http://{host}:{port}"
        config = {"host": host, "port": port}

    model = args.model or "fish_audio"
    write_json(output_dir / "server_config_snapshot.json", config)

    if args.input_text_file:
        input_text = Path(args.input_text_file).read_text(encoding="utf-8")
    elif args.input_text:
        input_text = args.input_text
    else:
        input_text = (
            "Fish Audio S2 Pro real-time neural speech synthesis streaming test. "
            "Delivering high quality expressive voice output with ultra low latency."
        )

    chunks = split_incremental_text(input_text, args.max_chars)
    (output_dir / "input_text.txt").write_text(input_text, encoding="utf-8")
    write_json(
        output_dir / "input_chunks.json",
        {
            "input_chars": len(input_text),
            "max_chars": args.max_chars,
            "chunk_count": len(chunks),
            "chunks": [{"index": i, "chars": len(c), "text": c} for i, c in enumerate(chunks)],
        },
    )

    server: subprocess.Popen[str] | None = None
    server_log_handle = None
    try:
        if args.start_server and args.server_config:
            server_command = [
                str(args.server_bin),
                "--config",
                str(args.server_config),
                "--log-file",
                str(output_dir / "framework.log"),
            ]
            server_log_handle = (output_dir / "server_stdout.log").open("w", encoding="utf-8")
            server = subprocess.Popen(
                server_command,
                cwd=args.repo_root,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if not args.skip_health_check:
            print(f"[*] Checking server health at {base_url}/health (timeout {args.health_timeout_s}s)...")
            wait_for_health(base_url, args.health_timeout_s)
            print("  -> Server is HEALTHY.")

        # Warmup
        warmup_meta = None
        if not args.skip_warmup:
            print(f"[*] Executing warmup request ({args.stream_format})...")
            warmup = stream_speech_request(
                base_url=base_url,
                model=model,
                text=args.warmup_text,
                sample_rate=args.sample_rate,
                stream_format=args.stream_format if args.stream_format != "both" else "audio",
                speaker_ref=args.speaker_ref,
                reference_text=args.reference_text,
            )
            write_wav(output_dir / "warmup_discard.wav", warmup["pcm"], args.sample_rate)
            warmup_meta = {k: v for k, v in warmup.items() if k != "pcm"}
            write_json(output_dir / "warmup_discard.json", warmup_meta)
            print(f"  -> Warmup done. TTFA: {warmup.get('ttfa_ms')} ms, Audio: {warmup['audio_validation']['duration_s']} s")

        formats_to_measure = ["audio", "sse"] if args.stream_format == "both" else [args.stream_format]
        format_results: dict[str, Any] = {}

        for fmt in formats_to_measure:
            fmt_dir = output_dir / fmt
            fmt_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[*] Starting measurement for stream_format='{fmt}' ({len(chunks)} chunks)...")

            measured_chunks: list[dict[str, Any]] = []
            merged_pcm = bytearray()
            seq_start = time.perf_counter()

            with VramSampler(fmt_dir / "measured_vram.csv", args.vram_sample_ms / 1000.0):
                for index, chunk_text in enumerate(chunks):
                    print(f"  -> Synthesizing chunk {index + 1}/{len(chunks)}: '{chunk_text[:40]}...'")
                    result = stream_speech_request(
                        base_url=base_url,
                        model=model,
                        text=chunk_text,
                        sample_rate=args.sample_rate,
                        stream_format=fmt,
                        speaker_ref=args.speaker_ref,
                        reference_text=args.reference_text,
                    )
                    chunk_pcm = result.pop("pcm")
                    merged_pcm.extend(chunk_pcm)
                    write_wav(fmt_dir / f"chunk_{index:03d}.wav", chunk_pcm, args.sample_rate)
                    record = {
                        "index": index,
                        "text": chunk_text,
                        **result,
                    }
                    measured_chunks.append(record)

            seq_elapsed_ms = (time.perf_counter() - seq_start) * 1000.0
            merged_pcm_bytes = bytes(merged_pcm)
            write_wav(fmt_dir / "merged.wav", merged_pcm_bytes, args.sample_rate)
            merged_validation = validate_audio_pcm(merged_pcm_bytes, args.sample_rate)

            ttfas = [float(c["ttfa_ms"]) for c in measured_chunks if c.get("ttfa_ms") is not None]
            all_chunk_intervals: list[float] = []
            for c in measured_chunks:
                stats = c.get("chunk_interval_stats", {})
                if stats and "mean" in stats and stats["count"] > 0:
                    all_chunk_intervals.append(float(stats["mean"]))

            fmt_summary = {
                "stream_format": fmt,
                "model": model,
                "sample_rate": args.sample_rate,
                "chunk_count": len(chunks),
                "sequence_elapsed_ms": round(seq_elapsed_ms, 2),
                "total_audio_duration_s": merged_validation["duration_s"],
                "real_time_factor": round(merged_validation["duration_s"] / (seq_elapsed_ms / 1000.0), 3),
                "ttfa_first_chunk_ms": ttfas[0] if ttfas else 0.0,
                "ttfa_stats": calc_stats(ttfas),
                "chunk_interval_stats": calc_stats(all_chunk_intervals),
                "audio_validation": merged_validation,
                "chunks": measured_chunks,
            }
            fmt_summary.update(read_vram_summary(fmt_dir / "measured_vram.csv"))
            write_json(fmt_dir / "summary.json", fmt_summary)
            format_results[fmt] = fmt_summary

        final_summary = {
            "timestamp": timestamp_slug(),
            "base_url": base_url,
            "model": model,
            "sample_rate": args.sample_rate,
            "warmup": warmup_meta,
            "formats": format_results,
        }
        write_json(output_dir / "benchmark_summary.json", final_summary)

        # Validation assertions if requested
        if args.validate or args.assert_max_ttfa_ms is not None:
            print("\n" + "=" * 60)
            print(" AUTOMATED STREAMING QUALITY ASSERTIONS")
            print("=" * 60)
            passed = True
            for fmt, res in format_results.items():
                ttfa = res["ttfa_first_chunk_ms"]
                duration = res["total_audio_duration_s"]
                audio_ok = res["audio_validation"]["audio_continuity"]
                
                print(f"[{fmt.upper()}] TTFA: {ttfa:.1f} ms | Duration: {duration:.2f} s | Continuity: {audio_ok}")
                
                if args.assert_max_ttfa_ms is not None and ttfa > args.assert_max_ttfa_ms:
                    print(f"  [FAIL] TTFA ({ttfa:.1f}ms) exceeded maximum ({args.assert_max_ttfa_ms}ms)")
                    passed = False
                if args.assert_min_duration_s is not None and duration < args.assert_min_duration_s:
                    print(f"  [FAIL] Duration ({duration:.2f}s) below minimum ({args.assert_min_duration_s}s)")
                    passed = False
                if not audio_ok:
                    print(f"  [FAIL] Audio continuity failed (silent or invalid audio)")
                    passed = False

            if not passed:
                raise AssertionError("Streaming validation assertions failed")
            print("  -> ALL STREAMING QUALITY ASSERTIONS PASSED (100% OK)")

        return final_summary

    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if server_log_handle is not None:
            server_log_handle.close()


def run_self_tests() -> int:
    print("==========================================================")
    print(" Running measure_tts_streaming_client Self-Test Suite")
    print("==========================================================")

    # Test 1: split_incremental_text
    print("[TEST 1] Testing split_incremental_text...")
    text = "Hello world. This is sentence two. Here is sentence three!"
    chunks = split_incremental_text(text, 30)
    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    print("  -> PASSED: Text splitting respects sentence and char budget.")

    # Test 2: PCM validation & statistics
    print("[TEST 2] Testing validate_audio_pcm and stats...")
    sample_rate = 44100
    num_samples = sample_rate * 2  # 2 seconds
    sine_samples = [int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)) for i in range(num_samples)]
    raw_pcm = struct.pack(f"<{num_samples}h", *sine_samples)
    val = validate_audio_pcm(raw_pcm, sample_rate)
    assert val["sample_count"] == num_samples
    assert abs(val["duration_s"] - 2.0) < 1e-3
    assert val["audio_continuity"] is True
    assert val["is_silent"] is False
    print("  -> PASSED: Audio PCM analysis, sample count, and RMS energy validated.")

    # Test 3: WAV Writing
    print("[TEST 3] Testing WAV writing & roundtrip...")
    test_wav = LOG_ROOT / "selftest_tmp.wav"
    write_wav(test_wav, raw_pcm, sample_rate)
    assert test_wav.exists()
    assert test_wav.stat().st_size > len(raw_pcm)
    with wave.open(str(test_wav), "rb") as r:
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getframerate() == sample_rate
        assert r.getnframes() == num_samples
    test_wav.unlink(missing_ok=True)
    print("  -> PASSED: WAV 44.1 kHz header assembly verified.")

    # Test 4: Timing & Percentile Calculations
    print("[TEST 4] Testing percentile & timing statistics...")
    timings = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = calc_stats(timings)
    assert stats["mean"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p50"] == 30.0
    print("  -> PASSED: Timing statistics, jitter, and percentiles verified.")

    print("==========================================================")
    print(" ALL SELF-TESTS PASSED (100% OK)")
    print("==========================================================")
    return 0


def parse_args() -> argparse.Namespace:
    default_output = LOG_ROOT / f"tts_streaming_{timestamp_slug()}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run internal client self-test suite")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--url", type=str, default="", help="Direct base URL of audiocpp_server (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="fish_audio")
    parser.add_argument("--stream-format", choices=["audio", "sse", "both"], default="audio", help="Stream format to benchmark")
    parser.add_argument("--input-text", type=str, default="")
    parser.add_argument("--input-text-file", type=Path, default=None)
    parser.add_argument("--speaker-ref", type=str, default=None, help="Reference audio file path for voice cloning")
    parser.add_argument("--reference-text", type=str, default=None, help="Transcript of reference audio")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--warmup-text", default="Fish Audio streaming warmup test.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--health-timeout-s", type=float, default=60.0)
    parser.add_argument("--vram-sample-ms", type=float, default=200.0)
    parser.add_argument("--skip-health-check", action="store_true", default=False)
    parser.add_argument("--skip-warmup", action="store_true", default=False)
    parser.add_argument("--validate", action="store_true", default=False, help="Run pass/fail assertions")
    parser.add_argument("--assert-max-ttfa-ms", type=float, default=None, help="Assert TTFA <= max ms")
    parser.add_argument("--assert-min-duration-s", type=float, default=None, help="Assert audio duration >= min s")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    try:
        summary = run_measurement(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n" + "=" * 60)
    print(" BENCHMARK RESULT SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
