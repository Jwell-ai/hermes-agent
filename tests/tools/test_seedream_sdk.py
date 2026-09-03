from unittest.mock import patch

from tools.seedream_sdk import create_seedream_image


def test_create_image_uses_internal_base_url_headers_and_options():
    captured = {}

    class FakeArk:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def post(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"object": "image.generation", "data": [{"url": "https://example.test/image.png"}]}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    with patch("tools.seedream_sdk._get_ark_class", return_value=FakeArk):
        result = create_seedream_image(
            base_url="http://jwell/internal/v1",
            headers={"X-App-Secret": "secret", "Idempotency-Key": "call-1"},
            model="seedream-4-0-250828",
            prompt="A warm autumn sunset",
            provider="byteplus",
            image_urls=["https://example.test/reference.png"],
            aspect_ratio="16:9",
            resolution="2K",
            quantity=2,
            watermark=True,
        )

    assert result == {
        "object": "image.generation",
        "data": [{"url": "https://example.test/image.png"}],
    }
    assert captured["path"] == "/images/generations"
    assert captured["client"]["base_url"].endswith("/internal/v1/")
    assert captured["client"]["http_client"].headers["X-App-Secret"] == "secret"
    assert captured["client"]["http_client"].headers["Idempotency-Key"] == "call-1"
    assert captured["kwargs"]["body"] == {
        "model": "seedream-4-0-250828",
        "prompt": "A warm autumn sunset",
        "image": ["https://example.test/reference.png"],
        "response_format": "url",
        "size": "2K",
        "provider": "byteplus",
        "aspect_ratio": "16:9",
        "n": 2,
        "watermark": True,
    }
