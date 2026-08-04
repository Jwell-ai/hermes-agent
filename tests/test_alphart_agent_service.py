"""Focused tests for shared Alphart agent service helpers."""

from alphart_agent_service import AlphartEduChatRequest, _alphart_enabled_toolsets, _media_intent, _provider_config_for_domain
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
