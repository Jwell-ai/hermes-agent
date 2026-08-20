"""Focused tests for shared Alphart agent service helpers."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import (
    _internal_relay_idempotency_key,
    _relay_request_overrides,
)

from alphart_agent_service import (
    AlphartEduChatRequest,
    AlphartEduTitleRequest,
    _alphart_enabled_toolsets,
    _canvas_video_recovery_needed,
    _canvas_workflow_item_type,
    _generation_tool_attempted,
    _generation_tool_effectively_failed,
    _forced_media_tool_messages,
    _media_intent,
    _post_chat_event_callback,
    _post_chat_result_callback,
    _provider_config_for_domain,
    _title_relay_idempotency_key,
    _uses_jwell_internal_relay,
)
from toolsets import resolve_toolset


def test_canvas_flat_multimodal_config_is_used():
    config = {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "endpoint": "https://example.test/v1beta/models/%s:generateContent",
        "api_key": "canvas-key",
        "timeout": 300,
    }

    resolved = _provider_config_for_domain(
        {"multimodal": config},
        "multimodal",
        {"provider": "gemini", "model": "gemini-3.5-flash"},
    )

    assert resolved == config


def test_canvas_toolsets_exclude_edu_workflows():
    request = AlphartEduChatRequest(app_scope="canvas")

    assert _alphart_enabled_toolsets(request) == ["alphart-canvas", "alphart-canvas-skills"]


def test_canvas_script_only_toolsets_exclude_edu_skill_management():
    request = AlphartEduChatRequest(app_scope="canvas", script_only=True)

    assert _alphart_enabled_toolsets(request) == ["alphart-canvas-skills"]


def test_edu_toolsets_are_explicit():
    request = AlphartEduChatRequest(app_scope="edu")

    assert _alphart_enabled_toolsets(request) == ["alphart-edu", "alphart-edu-skills"]


def test_jwell_billing_ownership_stays_edu_scoped(monkeypatch):
    monkeypatch.setenv("JWELL_SERVICE_GRPC_ADDR", "http://jwell.test")
    monkeypatch.setenv("JWELL_APP_SECRET", "secret")

    assert _uses_jwell_internal_relay(AlphartEduChatRequest(app_scope="edu")) is True
    assert _uses_jwell_internal_relay(AlphartEduChatRequest(app_scope="canvas")) is False


def test_jwell_callbacks_include_app_secret(monkeypatch):
    monkeypatch.setenv("JWELL_SERVICE_GRPC_ADDR", "http://jwell.test")
    monkeypatch.setenv("JWELL_APP_SECRET", "secret")
    monkeypatch.setenv("ALPHART_EDU_BACKEND_URL", "http://edu-backend")
    request = AlphartEduChatRequest(app_scope="edu", session_id="session-1")
    response = SimpleNamespace(status_code=200, text="{}")

    with patch("alphart_agent_service.requests.post", return_value=response) as post:
        _post_chat_result_callback(request, {"final_response": "done"})
        _post_chat_event_callback(request, {"type": "progress"})

    assert post.call_count == 2
    assert post.call_args_list[0].args[0] == "http://edu-backend/internal/api/v1/agent/chat-results"
    assert post.call_args_list[1].args[0] == "http://edu-backend/internal/api/v1/agent/events"
    for call in post.call_args_list:
        assert "X-App-Secret" not in call.kwargs["headers"]


def test_internal_relay_text_key_is_stable_for_retries_and_changes_per_turn():
    agent = SimpleNamespace(
        session_id="session-1",
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_mode="anthropic_messages",
        request_overrides={
            "extra_headers": {
                "X-App-Secret": "secret",
                "X-Internal-User-ID": "42",
            },
        },
    )
    first_turn = [{"role": "user", "content": "Make an image"}]
    second_turn = first_turn + [{"role": "assistant", "content": "Done"}]

    first_key = _internal_relay_idempotency_key(agent, first_turn)
    retry_key = _internal_relay_idempotency_key(agent, first_turn)
    second_key = _internal_relay_idempotency_key(agent, second_turn)
    overrides = _relay_request_overrides(agent, first_turn)

    assert first_key == retry_key
    assert first_key != second_key
    assert overrides["extra_headers"]["Idempotency-Key"] == first_key


def test_title_relay_key_is_scoped_to_user_and_session():
    request = AlphartEduTitleRequest(
        messages=[{"role": "user", "content": "Create a storybook"}],
        user_id="user-1",
        session_id="session-1",
    )
    same_request_key = _title_relay_idempotency_key(request, "openai", "gpt-5.4", "Create a storybook")
    other_session = request.model_copy(update={"session_id": "session-2"})
    other_user = request.model_copy(update={"user_id": "user-2"})

    assert same_request_key == _title_relay_idempotency_key(request, "openai", "gpt-5.4", "Create a storybook")
    assert same_request_key != _title_relay_idempotency_key(other_session, "openai", "gpt-5.4", "Create a storybook")
    assert same_request_key != _title_relay_idempotency_key(other_user, "openai", "gpt-5.4", "Create a storybook")


def test_edu_toolset_does_not_advertise_canvas_graph_mutations():
    edu_tools = set(resolve_toolset("alphart-edu"))

    assert {"canvas_create_node", "canvas_update_node", "canvas_connect_nodes"}.isdisjoint(edu_tools)


def test_canvas_toolset_owns_canvas_graph_mutations():
    canvas_tools = set(resolve_toolset("alphart-canvas"))

    assert {"canvas_create_node", "canvas_update_node", "canvas_connect_nodes"}.issubset(canvas_tools)


def test_prompt_refinement_does_not_route_to_media_generation():
    assert _media_intent(
        "refine prompt which describe a breathtaking autumn dusk landscape of Yellowstone National Park"
    ) == ""
    assert _media_intent("refine the prompt and generate an image of Yellowstone at dusk") == "image"


def test_canvas_explicit_video_request_overrides_selected_image():
    request = AlphartEduChatRequest(
        app_scope="canvas",
        selected_canvas_item_type="image",
        messages=[{
            "role": "user",
            "content": "[skill:video-cinematic-shot] @text node as prompt, @image node as keyframe, generate a 15s video",
        }],
    )

    assert _canvas_workflow_item_type(request) == "video"


def test_canvas_video_recovery_routes_after_speculative_image_failure():
    request = AlphartEduChatRequest(
        app_scope="canvas",
        messages=[{
            "role": "user",
            "content": "[skill:video-cinematic-shot] @text node as prompt, @image node as keyframe, generate a 15s video",
        }],
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "image-call", "function": {"name": "canvas_generate_image"}}],
        },
        {
            "role": "tool",
            "name": "canvas_generate_image",
            "content": '{"success": false, "error": "Canvas video request already has a video workflow"}',
        },
    ]

    assert _canvas_video_recovery_needed(request, messages) is True
    messages.append({"role": "assistant", "tool_calls": [{"id": "video-call", "function": {"name": "canvas_generate_video"}}]})
    assert _canvas_video_recovery_needed(request, messages) is False


def test_empty_audio_tool_call_remains_recoverable_until_scripted_call_succeeds():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "empty-audio",
                "function": {"name": "canvas_generate_audio", "arguments": '{"input":""}'},
            }],
        },
        {
            "role": "tool",
            "name": "canvas_generate_audio",
            "content": '{"success":false,"error":"audio input text is required"}',
        },
    ]

    assert _generation_tool_attempted(messages, "audio") is False

    messages.extend([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "scripted-audio",
                "function": {
                    "name": "canvas_generate_audio",
                    "arguments": '{"input":"A complete ready-to-speak lesson script."}',
                },
            }],
        },
        {
            "role": "tool",
            "name": "canvas_generate_audio",
            "content": '{"status":"success","result":{"type":"generate_audio_result"}}',
        },
    ])

    assert _generation_tool_attempted(messages, "audio") is True
    assert _generation_tool_effectively_failed(messages, "audio") is False


def test_scripted_audio_provider_failure_is_not_retried_as_missing_input():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "audio-provider-failure",
                "function": {
                    "name": "canvas_generate_audio",
                    "arguments": '{"input":"A complete ready-to-speak lesson script."}',
                },
            }],
        },
        {
            "role": "tool",
            "name": "canvas_generate_audio",
            "content": '{"success":false,"error":"provider returned HTTP 502"}',
        },
    ]

    assert _generation_tool_attempted(messages, "audio") is True
    assert _generation_tool_effectively_failed(messages, "audio") is True


def test_empty_audio_call_is_repaired_with_visible_script_then_audio():
    invalid_messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "empty-audio",
                "function": {"name": "canvas_generate_audio", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "name": "canvas_generate_audio",
            "content": '{"success":false,"error":"audio input text is required"}',
        },
    ]

    with patch(
        "alphart_agent_service._handle_alphart_generate_audio",
        return_value='{"status":"success","result":{"type":"generate_audio_result"}}',
    ) as generate_audio:
        repaired = _forced_media_tool_messages(
            "生成一段音频介绍万有引力",
            invalid_messages,
            invalid_messages,
        )

    script = repaired[0]["content"]
    tool_arguments = json.loads(repaired[1]["tool_calls"][0]["function"]["arguments"])
    assert script
    assert tool_arguments["input"] == script
    assert generate_audio.call_args.args[0]["input"] == script
    assert repaired[2]["role"] == "tool"
