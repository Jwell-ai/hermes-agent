"""Regression tests for the shared video relay payload."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from tools.alphart_tools import _backend_tool_timeout, _handle_alphart_generate_image, _handle_alphart_generate_video, alphart_context


def test_edu_video_relay_does_not_send_canvas_caption_script():
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
    assert "caption_script" not in captured["json"]


def test_video_relay_extracts_duration_from_user_message():
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
            "app_scope": "edu",
            "backend_url": "http://edu-backend",
            "user_message": "Generate a 5s 720p video with @image as keyframe",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 5


def test_video_relay_defaults_duration_to_five_seconds():
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
            "app_scope": "edu",
            "backend_url": "http://edu-backend",
            "user_message": "Generate a video with @image as keyframe",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 5


def test_video_relay_prefers_canvas_duration_option():
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
            "canvas_id": "canvas-1",
            "canvas_item_id": "video-1",
            "duration_seconds": 11,
            "user_message": "Generate a 5s video with @image as keyframe",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 11


def test_video_relay_prefers_llm_duration_over_canvas_option():
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
            "canvas_id": "canvas-1",
            "canvas_item_id": "video-1",
            "duration_seconds": 11,
            "user_message": "Generate a 5s video with @image as keyframe",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {
                    "prompt": "Animate the keyframe",
                    "provider": "peanut-video",
                    "model": "seedance",
                    "duration_seconds": 7,
                }
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 7


def test_video_relay_defaults_canvas_duration_without_option():
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
            "canvas_id": "canvas-1",
            "canvas_item_id": "video-1",
            "duration_seconds": 0,
            "user_message": "Generate a video with @image as keyframe",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["duration"] == 5


def test_seedance_video_relay_uses_native_jwell_route():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
            raise_for_status=lambda: None,
        )

    with alphart_context(
        {
            "app_scope": "edu",
            "backend_url": "http://jwell-relay",
            "application_backend_url": "http://edu-backend",
        }
    ), patch.dict(
        os.environ,
        {"JWELL_SERVICE_GRPC_ADDRS": "http://jwell-relay", "JWELL_APP_SECRET": "secret"},
    ), patch("tools.seedance_sdk.create_seedance_task", return_value={"id": "task-1", "status": "queued"}) as sdk_call, patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "byteplus", "model": "dreamina-seedance-2-0"},
                tool_call_id="call-seedance-1",
            )
        )

    assert result["status"] == "success"
    assert sdk_call.call_args.kwargs["base_url"] == "http://jwell-relay/internal/v3"
    assert sdk_call.call_args.kwargs["headers"]["Idempotency-Key"] == "call-seedance-1"
    assert captured["url"] == "http://edu-backend/internal/api/v1/agent/jwell-video-tasks"


def test_doubao_seedance_video_relay_uses_native_jwell_route():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
            raise_for_status=lambda: None,
        )

    with alphart_context({"app_scope": "edu", "backend_url": "http://edu-backend"}), patch.dict(
        os.environ,
        {"JWELL_SERVICE_GRPC_ADDRS": "http://jwell-relay", "JWELL_APP_SECRET": "secret"},
    ), patch("tools.seedance_sdk.create_seedance_task", return_value={"id": "task-1", "status": "queued"}) as sdk_call, patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "byteplus", "model": "doubao-seedance-2-0-260128"}
            )
        )

    assert result["status"] == "success"
    assert sdk_call.call_args.kwargs["base_url"] == "http://jwell-relay/internal/v3"
    assert captured["url"] == "http://edu-backend/internal/api/v1/agent/jwell-video-tasks"


def test_canvas_video_relay_uses_local_task_id_when_provider_id_is_not_ready():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return SimpleNamespace(
            status_code=202,
            text='{"status":"processing","task_id":"canvas-task-1","canvas_task_id":"canvas-task-1"}',
            json=lambda: {
                "status": "processing",
                "task_id": "canvas-task-1",
                "canvas_task_id": "canvas-task-1",
            },
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "canvas_item_id": "video-1",
            "canvas_item_type": "video",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "other-video"}
            )
        )

    assert result["status"] == "success"
    assert result["result"]["task_id"] == "canvas-task-1"
    assert captured["url"] == "http://canvas-backend/internal/v1/videos"


def test_canvas_video_relay_keeps_canvas_caption_script():
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
            "canvas_item_id": "video-node",
            "video_caption_script": "Keep walking toward the sunrise.",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "A cinematic sunrise", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
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
                {"prompt": "A warm autumn sunset", "provider": "peanut-image-gpt", "model": "gpt-image-2"},
                tool_call_id="image-call-1",
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["canvas_item_id"] == "image-node"
    assert captured["json"]["tool_call_id"] == "image-call-1"


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


def test_edu_seedream_image_uses_ark_sdk_through_jwell_relay():
    with alphart_context(
        {
            "app_scope": "edu",
            "backend_url": "http://edu-backend",
        }
    ), patch.dict(
        os.environ,
        {
            "JWELL_SERVICE_GRPC_ADDRS": "http://jwell-relay",
            "JWELL_APP_SECRET": "secret",
        },
    ), patch(
        "tools.seedream_sdk.create_seedream_image",
        return_value={"data": [{"url": "https://jwell/image.png", "model": "seedream-4-0"}]},
    ) as sdk_call, patch(
        "tools.alphart_tools.requests.post"
    ) as post_call, patch(
        "tools.alphart_tools._resolve_jwell_media_urls",
        return_value=["https://jwell/reference.png"],
    ), patch(
        "tools.alphart_tools._import_jwell_media",
        return_value={"url": "https://jwell/image.png", "s3_object_name": "test/image.png"},
    ):
        result = json.loads(
            _handle_alphart_generate_image(
                {
                    "prompt": "A warm autumn sunset",
                    "provider": "byteplus",
                    "model": "seedream-4-0-250828",
                    "input_images": ["https://example.test/reference.png"],
                    "aspect_ratio": "16:9",
                    "resolution": "2K",
                },
                tool_call_id="call-seedream-1",
            )
        )

    assert result["status"] == "success"
    assert result["result"]["s3_object_name"] == "test/image.png"
    assert sdk_call.call_args.kwargs["base_url"] == "http://jwell-relay/internal/v1"
    assert sdk_call.call_args.kwargs["headers"]["X-App-Secret"] == "secret"
    assert sdk_call.call_args.kwargs["provider"] == "byteplus"
    assert sdk_call.call_args.kwargs["idempotency_key"] == "call-seedream-1"
    post_call.assert_not_called()


def test_canvas_video_request_does_not_generate_speculative_image():
    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "user_message": "[skill:video-cinematic-shot] @text node as prompt, @image node as keyframe, generate a 15s video",
        }
    ), patch("tools.alphart_tools.requests.post") as post:
        result = json.loads(
            _handle_alphart_generate_image(
                {"prompt": "A speculative keyframe", "provider": "peanut-image-gemini", "model": "gemini-image"}
            )
        )

    assert result["success"] is False
    assert "video workflow" in result["error"]
    post.assert_not_called()


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
            "canvas_item_id": "video-node",
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


def test_canvas_video_relay_targets_new_graph_video_node():
    captured = {}
    created = []
    created_bodies = []

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/internal/api/v1/canvas/nodes"):
            item_type = body["item_type"]
            item_id = f"{item_type}-node"
            created.append(item_type)
            created_bodies.append(body)
            return SimpleNamespace(
                status_code=201,
                text=json.dumps({"item": {"id": item_id}}),
                json=lambda: {"item": {"id": item_id}},
            )
        captured["json"] = body
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the scene", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert created == ["text", "video"]
    assert captured["json"]["canvas_item_id"] == "video-node"
    assert captured["json"]["create_if_missing"] is True


def test_canvas_video_relay_ignores_stale_model_target_and_creates_graph():
    captured = {}
    created = []

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/internal/api/v1/canvas/nodes"):
            item_type = body["item_type"]
            item_id = f"{item_type}-node"
            created.append(item_type)
            return SimpleNamespace(
                status_code=201,
                text=json.dumps({"item": {"id": item_id}}),
                json=lambda: {"item": {"id": item_id}},
            )
        captured["json"] = body
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {
                    "canvas_item_id": "stale-model-target",
                    "prompt": "Animate the scene",
                    "provider": "peanut-video",
                    "model": "seedance",
                }
            )
        )

    assert result["status"] == "success"
    assert created == ["text", "video"]
    assert captured["json"]["canvas_item_id"] == "video-node"
    assert captured["json"]["create_if_missing"] is True


def test_canvas_video_relay_does_not_use_selected_text_node_as_video_target():
    captured = {}
    created = []

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/internal/api/v1/canvas/nodes"):
            item_type = body["item_type"]
            item_id = f"{item_type}-node"
            created.append(item_type)
            return SimpleNamespace(
                status_code=201,
                text=json.dumps({"item": {"id": item_id}}),
                json=lambda: {"item": {"id": item_id}},
            )
        captured["json"] = body
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "canvas_item_id": "text-node",
            "canvas_item_type": "text",
            "selected_canvas_item_id": "text-node",
            "selected_canvas_item_type": "text",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {
                    "prompt": "Animate the scene",
                    "provider": "peanut-video",
                    "model": "seedance",
                }
            )
        )

    assert result["status"] == "success"
    assert created == ["text", "video"]
    assert captured["json"]["canvas_item_id"] == "video-node"


def test_canvas_video_graph_keeps_references_on_output_only():
    captured = {}
    created_bodies = []

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/internal/api/v1/canvas/nodes"):
            created_bodies.append(body)
            item_type = body["item_type"]
            return SimpleNamespace(
                status_code=201,
                text=json.dumps({"item": {"id": f"{item_type}-node"}}),
                json=lambda item_type=item_type: {"item": {"id": f"{item_type}-node"}},
            )
        captured["json"] = body
        return SimpleNamespace(status_code=202, text='{"id":"task-1"}', json=lambda: {"id": "task-1"})

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
            "reference_item_ids": ["text-ref", "image-ref"],
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the scene", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert "source_item_ids" not in created_bodies[0]
    assert created_bodies[1]["source_item_ids"] == ["text-ref", "image-ref", "text-node"]
    assert captured["json"]["canvas_item_id"] == "video-node"


def test_canvas_timeout_does_not_inherit_edu_only_value():
    with patch.dict(os.environ, {"ALPHART_EDU_BACKEND_TOOL_TIMEOUT_SECONDS": "180"}, clear=True):
        with alphart_context({"app_scope": "canvas"}):
            assert _backend_tool_timeout() == 900
        with alphart_context({"app_scope": "edu"}):
            assert _backend_tool_timeout() == 180


def test_canvas_video_relay_creates_fresh_graph_for_each_automatic_request():
    captured = []
    created = []
    node_number = {"text": 0, "video": 0}

    def fake_post(url, **kwargs):
        body = kwargs["json"]
        if url.endswith("/internal/api/v1/canvas/nodes"):
            item_type = body["item_type"]
            node_number[item_type] += 1
            item_id = f"{item_type}-node-{node_number[item_type]}"
            created.append((item_type, item_id))
            return SimpleNamespace(
                status_code=201,
                text=json.dumps({"item": {"id": item_id}}),
                json=lambda item_id=item_id: {"item": {"id": item_id}},
            )
        captured.append(body)
        return SimpleNamespace(
            status_code=202,
            text='{"id":"task-1","status":"queued"}',
            json=lambda: {"id": "task-1", "status": "queued"},
        )

    with alphart_context(
        {
            "app_scope": "canvas",
            "backend_url": "http://canvas-backend",
            "canvas_id": "canvas-1",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        first = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the first scene", "provider": "peanut-video", "model": "seedance"}
            )
        )
        second = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the second scene", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert created == [
        ("text", "text-node-1"),
        ("video", "video-node-1"),
        ("text", "text-node-2"),
        ("video", "video-node-2"),
    ]
    assert [item["canvas_item_id"] for item in captured] == ["video-node-1", "video-node-2"]


def test_canvas_video_relay_uses_selected_video_node():
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
            "canvas_id": "canvas-1",
            "selected_canvas_item_id": "video-node",
            "selected_canvas_item_type": "video",
        }
    ), patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the scene", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["status"] == "success"
    assert captured["json"]["canvas_item_id"] == "video-node"


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
            "canvas_item_id": "video-node",
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
            "canvas_item_id": "video-node",
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
    with alphart_context(
        {"app_scope": "canvas", "backend_url": "http://canvas-backend", "canvas_item_id": "video-node"}
    ), patch(
        "tools.alphart_tools.requests.post", return_value=response
    ):
        result = json.loads(
            _handle_alphart_generate_video(
                {"prompt": "Animate the keyframe", "provider": "peanut-video", "model": "seedance"}
            )
        )

    assert result["success"] is False
    assert "video keyframe object not found" in result["error"]
