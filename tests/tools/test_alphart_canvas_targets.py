from tools.alphart_tools import (
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
