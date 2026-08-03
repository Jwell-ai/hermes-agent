"""Focused tests for shared Alphart agent service helpers."""

from alphart_agent_service import _provider_config_for_domain


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
