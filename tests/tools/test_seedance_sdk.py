from unittest.mock import patch

import pytest

from tools.lazy_deps import FeatureUnavailable
from tools.seedance_sdk import _get_ark_class, _native_content, create_seedance_task


def test_native_content_assigns_first_frame_role_to_single_image():
    assert _native_content("Animate this", ["https://example.test/a.png"], []) == [
        {"type": "text", "text": "Animate this"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/a.png"},
            "role": "first_frame",
        },
    ]


def test_native_content_assigns_reference_role_for_multiple_images():
    content = _native_content(
        "Animate these",
        ["https://example.test/a.png", "https://example.test/b.png"],
        [],
        ["first_frame", ""],
    )
    assert content[1]["role"] == "first_frame"
    assert content[2]["role"] == "reference_image"


def test_native_content_keeps_unlabelled_multiple_images_as_references():
    content = _native_content(
        "Use both as visual references",
        ["https://example.test/a.png", "https://example.test/b.png"],
        [],
    )
    assert [item["role"] for item in content[1:]] == [
        "reference_image",
        "reference_image",
    ]


def test_native_content_normalizes_user_facing_image_roles():
    content = _native_content(
        "Animate this",
        ["https://example.test/keyframe.jfif"],
        [],
        ["keyframe"],
    )
    assert content[1]["role"] == "first_frame"

    content = _native_content(
        "Use this as a visual reference",
        ["https://example.test/reference.jfif"],
        [],
        ["protagonist"],
    )
    assert content[1]["role"] == "first_frame"


def test_native_content_assigns_soundtrack_role_for_audio():
    content = _native_content("Animate with audio", [], ["https://example.test/a.wav"])
    assert content[-1] == {
        "type": "audio_url",
        "audio_url": {"url": "https://example.test/a.wav"},
        "role": "soundtrack",
    }


def test_native_content_defaults_blank_audio_role_to_soundtrack():
    content = _native_content(
        "Animate with audio",
        [],
        ["https://example.test/a.wav"],
        audio_roles=[""],
    )
    assert content[-1]["role"] == "soundtrack"


def test_get_ark_class_preserves_lazy_install_failure():
    failure = FeatureUnavailable(
        "video.seedance",
        ("volcengine-python-sdk[ark]",),
        "lazy installs disabled",
    )
    with patch("tools.lazy_deps.ensure", side_effect=failure), pytest.raises(
        RuntimeError,
        match="lazy installs disabled",
    ):
        _get_ark_class()


def test_create_task_uses_internal_base_url_and_headers():
    captured = {}

    class FakeTask:
        def to_dict(self, **_kwargs):
            return {"id": "relay-request-1", "status": "queued"}

    class FakeTasks:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeTask()

    class FakeContentGeneration:
        tasks = FakeTasks()

    class FakeArk:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.content_generation = FakeContentGeneration()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    with patch("tools.seedance_sdk._get_ark_class", return_value=FakeArk):
        result = create_seedance_task(
            base_url="http://jwell/internal/v3",
            headers={"X-App-Secret": "secret", "Idempotency-Key": "call-1"},
            model="doubao-seedance-2-0-260128",
            prompt="Animate the keyframe",
            image_urls=["https://example.test/keyframe.png"],
            ratio="16:9",
            resolution="720p",
            duration=5,
            generate_audio=False,
        )

    assert result == {"id": "relay-request-1", "status": "queued"}
    assert captured["client"]["base_url"] == "http://jwell/internal/v3/"
    assert captured["client"]["http_client"].headers["X-App-Secret"] == "secret"
    assert captured["client"]["http_client"].headers["Idempotency-Key"] == "call-1"
    assert captured["kwargs"]["model"] == "doubao-seedance-2-0-260128"
    assert captured["kwargs"]["ratio"] == "16:9"
    assert captured["kwargs"]["content"][1]["image_url"]["url"].endswith("keyframe.png")
