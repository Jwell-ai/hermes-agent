import json
from unittest.mock import patch

from tools.alphart_tools import (
    _handle_canvas_create_node,
    _canvas_trusted_audio_target,
    _canvas_trusted_image_target,
    _canvas_trusted_video_target,
    alphart_context,
)


def test_canvas_media_targets_accept_explicitly_mentioned_nodes():
    mentioned_id = "mentioned-node"
    with alphart_context({
        "app_scope": "canvas",
        "mentioned_node_ids": [mentioned_id],
        "selected_canvas_item_id": "selected-node",
        "selected_canvas_item_type": "image",
    }):
        assert _canvas_trusted_image_target(mentioned_id) == mentioned_id
        assert _canvas_trusted_audio_target(mentioned_id) == mentioned_id
        assert _canvas_trusted_video_target(mentioned_id) == mentioned_id


def test_edu_does_not_trust_canvas_mentioned_node_context():
    mentioned_id = "mentioned-node"
    with alphart_context({
        "app_scope": "edu",
        "mentioned_node_ids": [mentioned_id],
    }):
        assert _canvas_trusted_image_target(mentioned_id) == ""
        assert _canvas_trusted_audio_target(mentioned_id) == ""
        assert _canvas_trusted_video_target(mentioned_id) == ""


def test_video_skill_uses_referenced_previs_and_captures_as_video_sources():
    source_ids = ["previs-1", "frame-1", "frame-2"]
    with (
        alphart_context({
            "app_scope": "canvas",
            "canvas_id": "canvas-1",
            "selected_canvas_item_id": "previs-1",
            "selected_canvas_item_type": "previs",
            "reference_item_ids": source_ids,
            "requested_action": "create_video",
            "requested_node_type": "video",
            "user_message": "[skill:video-cinematic-shot] Use @previs and captured keyframes",
        }),
        patch("tools.alphart_tools._internal_api_url", return_value="http://canvas/internal"),
        patch("tools.alphart_tools._internal_relay_headers", return_value={}),
        patch("tools.alphart_tools.requests.post") as post,
    ):
        post.return_value.status_code = 201
        post.return_value.json.return_value = {"canvas_item_id": "video-1"}
        video_result = json.loads(_handle_canvas_create_node({
            "item_type": "video",
            "source_item_ids": source_ids,
        }))

    assert video_result["status"] == "success"
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["source_item_ids"] == source_ids


def test_video_skill_can_create_explicit_previs_then_video():
    with (
        alphart_context({
            "app_scope": "canvas",
            "canvas_id": "canvas-1",
            "requested_action": "create_video",
            "requested_node_type": "video",
            "user_message": "[skill:video-cinematic-shot] Create a separate 3D previs scene, then generate a video from it",
        }),
        patch("tools.alphart_tools._internal_api_url", return_value="http://canvas/internal"),
        patch("tools.alphart_tools._internal_relay_headers", return_value={}),
        patch("tools.alphart_tools.requests.post") as post,
    ):
        post.return_value.status_code = 201
        post.return_value.json.side_effect = [
            {"canvas_item_id": "previs-1"},
            {"canvas_item_id": "video-1"},
        ]
        previs_result = json.loads(_handle_canvas_create_node({
            "item_type": "previs",
            "title": "Shot",
            "content": {"previs_scene": {"version": 1}},
        }))
        video_result = json.loads(_handle_canvas_create_node({
            "item_type": "video",
            "title": "Rendered shot",
            "source_item_ids": ["previs-1"],
        }))

    assert previs_result["status"] == "success"
    assert video_result["status"] == "success"
    assert post.call_count == 2
    assert post.call_args_list[0].kwargs["json"]["item_type"] == "previs"
    assert post.call_args_list[1].kwargs["json"]["source_item_ids"] == ["previs-1"]
