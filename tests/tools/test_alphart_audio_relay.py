"""Tests for long-script audio relay chunking and concatenation."""

import base64
import io
import json
import wave
from unittest.mock import MagicMock, patch

import requests

from tools.alphart_tools import (
    _generate_chunked_audio,
    _import_jwell_media,
    _relay_headers,
    _split_audio_script,
    alphart_context,
)


def _wav_bytes(frames: int = 100) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def test_relay_headers_include_media_idempotency_key():
    with alphart_context({
        "app_scope": "edu",
        "user_id": "42",
        "user_uuid": "user-uuid",
        "org_no": "org-1",
        "session_id": "session-1",
    }):
        headers = _relay_headers("audio-call:chunk:2")

    assert headers["X-Internal-User-ID"] == "42"
    assert headers["X-Internal-User-UUID"] == "user-uuid"
    assert headers["X-Org-No"] == "org-1"
    assert headers["Idempotency-Key"] == "audio-call:chunk:2"


def test_split_audio_script_targets_natural_english_chunks():
    text = " ".join(f"word{index}." for index in range(150))

    chunks = _split_audio_script(text, "english")

    assert len(chunks) == 2
    assert all(chunk for chunk in chunks)
    assert "word0" in chunks[0]
    assert "word149" in chunks[1]


def test_chunked_audio_retries_each_chunk_and_concatenates_wav():
    chunks = ["First sentence. " + "first " * 80, "Second sentence. " + "second " * 80]
    wav = base64.b64encode(_wav_bytes(100)).decode("ascii")
    calls = []

    def fake_generate(args):
        calls.append(dict(args))
        if len(calls) == 2:
            return '{"success":false,"error":"temporary provider failure"}'
        return json.dumps({
            "status": "success",
            "result": {
                "type": "generate_audio_result",
                "url": f"data:audio/wav;base64,{wav}",
                "mime_type": "audio/wav",
                "usage": {"total_tokens": 10},
            },
        })

    with patch("tools.alphart_tools._handle_alphart_generate_audio", side_effect=fake_generate), \
        patch("tools.alphart_tools.time.sleep"):
        result = json.loads(_generate_chunked_audio(
            {
                "input": "the complete script",
                "provider": "google",
                "model": "gemini-3.1-flash-tts-preview",
                "language_type": "english",
            },
            "the complete script",
            chunks,
            tool_call_id="audio-call",
        ))

    assert result["status"] == "success"
    payload = result["result"]
    assert payload["chunk_count"] == 2
    assert payload["generated_chunk_count"] == 2
    assert payload["usage"]["total_tokens"] == 20
    assert [call["tool_call_id"] for call in calls] == [
        "audio-call:chunk:1",
        "audio-call:chunk:2",
        "audio-call:chunk:2",
    ]
    combined = base64.b64decode(payload["url"].split(",", 1)[1])
    with wave.open(io.BytesIO(combined), "rb") as audio:
        assert audio.getnframes() == 200 + int(24000 * 0.35)


def test_chunked_audio_reports_partial_failure():
    with patch(
        "tools.alphart_tools._handle_alphart_generate_audio",
        return_value='{"success":false,"error":"provider unavailable"}',
    ), patch("tools.alphart_tools.time.sleep"):
        result = json.loads(_generate_chunked_audio(
            {
                "input": "the complete script",
                "provider": "openai",
                "model": "gpt-4o-mini-tts",
                "tool_call_id": "audio-call",
            },
            "the complete script",
            ["first", "second"],
        ))

    assert result["status"] == "failed"
    assert result["result"]["status"] == "partial"
    assert result["result"]["failed_chunk_index"] == 1
    assert result["result"]["generated_chunk_count"] == 0


def test_chunked_audio_counts_generated_chunk_when_download_fails():
    wav = base64.b64encode(_wav_bytes()).decode("ascii")
    calls = 0

    def fake_generate(_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({
                "status": "success",
                "result": {
                    "url": f"data:audio/wav;base64,{wav}",
                    "usage": {"total_tokens": 10},
                },
            })
        return json.dumps({
            "status": "success",
            "result": {
                "url": "https://audio.invalid/missing.wav",
                "usage": {"total_tokens": 10},
            },
        })

    with patch("tools.alphart_tools._handle_alphart_generate_audio", side_effect=fake_generate), \
        patch("tools.alphart_tools.requests.get", side_effect=requests.RequestException("download failed")), \
        patch("tools.alphart_tools.time.sleep"):
        result = json.loads(_generate_chunked_audio(
            {"provider": "openai", "model": "gpt-4o-mini-tts", "tool_call_id": "audio-call"},
            "the complete script",
            ["first", "second"],
        ))

    assert result["status"] == "failed"
    assert result["result"]["failed_chunk_index"] == 2
    assert result["result"]["generated_chunk_count"] == 1


def test_chunked_audio_retries_final_import_with_stable_object_name():
    wav = base64.b64encode(_wav_bytes()).decode("ascii")
    imports = []

    def fake_import(asset, media_type, object_name=""):
        imports.append((asset, media_type, object_name))
        if len(imports) == 1:
            raise requests.RequestException("temporary storage failure")
        return {
            **asset,
            "url": "https://storage.example/audio.wav",
            "s3_object_name": "org/audio.wav",
            "mime_type": "audio/wav",
        }

    with patch(
        "tools.alphart_tools._handle_alphart_generate_audio",
        return_value=json.dumps({
            "status": "success",
            "result": {
                "url": f"data:audio/wav;base64,{wav}",
                "audio_url": f"data:audio/wav;base64,{wav}",
            },
        }),
    ), patch("tools.alphart_tools._jwell_relay_enabled", return_value=True), \
        patch("tools.alphart_tools._import_jwell_media", side_effect=fake_import), \
        patch("tools.alphart_tools.time.sleep"):
        result = json.loads(_generate_chunked_audio(
            {"provider": "google", "model": "gemini-3.1-flash-tts-preview"},
            "the complete script",
            ["first", "second"],
            tool_call_id="audio-call",
        ))

    assert result["status"] == "success"
    assert len(imports) == 2
    assert imports[0][1] == imports[1][1] == "audio"
    assert imports[0][2] == imports[1][2]
    assert imports[0][2].startswith("audio-")
    assert imports[0][2].endswith(".wav")
    assert result["result"]["url"] == "https://storage.example/audio.wav"
    assert result["result"]["audio_url"] == "https://storage.example/audio.wav"
    assert not result["result"]["audio_url"].startswith("data:")


def test_import_jwell_media_replaces_provider_audio_alias_with_persistent_url():
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "url": "https://storage.example/audio.wav",
            "s3_object_name": "org/audio.wav",
        },
    }

    with patch("tools.alphart_tools._jwell_relay_enabled", return_value=True), \
        patch("tools.alphart_tools._internal_api_url", return_value="http://edu/internal/import"), \
        patch("tools.alphart_tools.requests.post", return_value=response):
        result = _import_jwell_media(
            {
                "url": "data:audio/wav;base64,AAAA",
                "audio_url": "data:audio/wav;base64,AAAA",
                "mime_type": "audio/wav",
            },
            "audio",
        )

    assert result["url"] == "https://storage.example/audio.wav"
    assert result["audio_url"] == "https://storage.example/audio.wav"
