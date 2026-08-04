"""Regression tests for the shared video relay payload."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.alphart_tools import _handle_alphart_generate_image, _handle_alphart_generate_video, alphart_context


def test_edu_video_relay_preserves_context_caption_script():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "edu",
            "backend_url": "http://edu-backend",
            "video_caption_script": "Keep walking toward the sunrise.",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "A cinematic sunrise", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["url"] == "http://edu-backend/internal/v1/videos"
    assert captured["json"]["caption_script"] == "Keep walking toward the sunrise."


def test_canvas_image_relay_targets_newly_created_image_node():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=200,
            text='{"data":[{"url":"https://canvas.test/image.png","s3_object_name":"org/doc/image.png"}]}',
            json=lambda: {"data": [{"url": "https://canvas.test/image.png", "s3_object_name": "org/doc/image.png"}]},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "_canvas_created_nodes": [{"id": "prompt-node", "item_type": "text"}, {"id": "image-node", "item_type": "image"}],
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_image(
                {"prompt": "A warm autumn sunset", "provider": "peanut-image-gpt", "model": "gpt-image-2"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["canvas_item_id"] == "image-node"


def test_canvas_image_relay_without_asset_is_failure():
    response = SimpleNamespace(status_code=200, text='{"data":[]}', json=lambda: {"data": []})
    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "_canvas_created_nodes": [
                {"id": "prompt-node", "item_type": "text"},
                {"id": "image-node", "item_type": "image"},
            ],
        }
    ), patch(
        "tools.alphart_tools.requests.post", return_value=response
    ):
        result = json.loads(
            _handle_alphart_generate_image(
                {"prompt": "A warm autumn sunset", "provider": "peanut-image-gpt", "model": "gpt-image-2"}
            )
        )

    assert result["success"] is False
    assert "no stored image asset" in result["error"]


def test_canvas_image_relay_accepts_object_key_asset():
    response = SimpleNamespace(
        status_code=200,
        text='{"data":[{"object_key":"org/doc/image.png"}]}',
        json=lambda: {"data": [{"object_key": "org/doc/image.png"}]},
    )
    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "_canvas_created_nodes": [
                {"id": "prompt-node", "item_type": "text"},
                {"id": "image-node", "item_type": "image"},
            ],
        }
    ), patch(
        "tools.alphart_tools.requests.post", return_value=response
    ):
        result = json.loads(
            _handle_alphart_generate_image(
                {"prompt": "A warm autumn sunset", "provider": "peanut-image-gpt", "model": "gpt-image-2"}
            )
        )

    assert result["status"] == "success"
    assert result["result"]["s3_object_name"] == "org/doc/image.png"


def test_canvas_video_relay_drops_model_supplied_url_references():
    captured = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    connected = {"s3_object_name": "org/doc/keyframe.png", "reference_note": "Keyframe"}
    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "input_images": [connected],
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {
                    "prompt": "Animate the keyframe",
                    "provider": "peanut-video",
                    "model": "seedance",
                    "input_images": [{"url": "https://provider-inaccessible.example/keyframe.png"}],
                }
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["image"] == [connected]


def test_canvas_video_prompt_audio_preference_overrides_ui_fallback():
    captured = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "user_message": "Generate a 10s video without audio",
            "generate_audio": True,
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the scene", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["generate_audio"] is False


def test_canvas_video_prompt_options_override_canvas_context():
    captured = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "duration_seconds": 5,
            "aspect_ratio": "16:9",
            "resolution": "720p",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {
                    "prompt": "Animate the scene",
                    "provider": "peanut-video",
                    "model": "seedance",
                    "duration_seconds": 10,
                    "aspect_ratio": "9:16",
                    "resolution": "1080p",
                }
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 10
    assert captured["json"]["aspect_ratio"] == "9:16"
    assert captured["json"]["resolution"] == "1080p"


def test_canvas_video_relay_returns_relay_error_instead_of_generic_retry():
    response = SimpleNamespace(
        status_code=400,
        text='{"detail":"video keyframe object not found"}',
        json=lambda: {"detail": "video keyframe object not found"},
    )
    with alphart_context({"app_scope": "canvas", "backend_url": "http://canvas-backend"}), patch(
        "tools.alphart_tools.requests.post", return_value=response
    ):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["success"] is False
    assert "video keyframe object not found" in result["error"]
