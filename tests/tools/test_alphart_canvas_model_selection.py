from tools.alphart_tools import (
    _apply_canvas_selected_media_model,
    alphart_context,
)


def test_canvas_selected_media_model_overrides_agent_tool_choice():
    args = {
        "id": "generate_video_by_other",
        "tool_id": "other-tool",
        "provider": "other-provider",
        "model": "other-model",
        "video_provider": "other-provider",
        "video_model": "other-model",
    }

    with alphart_context({"app_scope": "canvas", "video_model": "byteplus:seedance-1-5-pro"}):
        _apply_canvas_selected_media_model(args, "video")

    assert args == {
        "provider": "byteplus",
        "model": "seedance-1-5-pro",
    }


def test_canvas_bare_media_model_discards_agent_provider():
    args = {"provider": "other-provider", "model": "other-model"}

    with alphart_context({"app_scope": "canvas", "image_model": "seedream-4-5"}):
        _apply_canvas_selected_media_model(args, "image")

    assert args == {"model": "seedream-4-5"}


def test_canvas_audio_settings_do_not_depend_on_catalog_order_without_model_choice():
    first_tool = {
        "type": "audio",
        "provider": "openai",
        "model": "other-tts",
    }
    selected_tool = {
        "type": "audio",
        "provider": "tencent",
        "model": "selected-tts",
    }

    for tool_list in ([first_tool, selected_tool], [selected_tool, first_tool]):
        args = {"voice": "selected-model-voice", "speed": 1.5}
        with alphart_context({
            "app_scope": "canvas",
            "audio_model": "tencent:selected-tts",
            "model_catalog_authoritative": True,
            "tool_list": tool_list,
        }):
            _apply_canvas_selected_media_model(args, "audio")

        assert args == {
            "voice": "selected-model-voice",
            "speed": 1.5,
            "provider": "tencent",
            "model": "selected-tts",
        }


def test_edu_media_model_selection_is_unchanged():
    args = {
        "id": "generate_audio_by_agent",
        "tool_id": "agent-tool",
        "provider": "agent-provider",
        "model": "agent-model",
        "audio_provider": "agent-provider",
        "audio_model": "agent-model",
    }

    with alphart_context({"app_scope": "edu", "audio_model": "frontend:selected"}):
        _apply_canvas_selected_media_model(args, "audio")

    assert args == {
        "id": "generate_audio_by_agent",
        "tool_id": "agent-tool",
        "provider": "agent-provider",
        "model": "agent-model",
        "audio_provider": "agent-provider",
        "audio_model": "agent-model",
    }
