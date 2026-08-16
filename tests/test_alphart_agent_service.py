"""Focused tests for shared Alphart agent service helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from alphart_agent_service import AlphartEduChatRequest, _alphart_enabled_toolsets, _canvas_video_recovery_needed, _canvas_workflow_item_type, _media_intent, _post_chat_event_callback, _post_chat_result_callback, _provider_config_for_domain, _uses_jwell_internal_relay
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
    request = AlphartEduChatRequest(app_scope="edu", session_id="session-1")
    response = SimpleNamespace(status_code=200, text="{}")

    with patch("alphart_agent_service.requests.post", return_value=response) as post:
        _post_chat_result_callback(request, {"final_response": "done"})
        _post_chat_event_callback(request, {"type": "progress"})

    assert post.call_count == 2
    for call in post.call_args_list:
        assert call.kwargs["headers"]["X-App-Secret"] == "secret"


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
