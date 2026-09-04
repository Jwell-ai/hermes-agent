"""Official Ark SDK bridge for Seedream image requests.

The client is pointed at Jwell's internal OpenAI-compatible relay. Jwell
continues to own model selection, identity, credits, persistence, and media
import for Alphart application requests.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tools.seedance_sdk import _get_ark_class


def _image_size(resolution: Any) -> str | None:
    value = str(resolution or "").strip().upper()
    return value if value in {"1K", "2K", "3K", "4K"} else None


def create_seedream_image(
    *,
    base_url: str,
    headers: Mapping[str, str],
    model: str,
    prompt: str,
    provider: str = "",
    image_urls: Iterable[str] = (),
    aspect_ratio: str = "",
    resolution: str = "",
    watermark: bool | None = None,
    quantity: int | None = None,
    watermark: bool | None = None,
    idempotency_key: str = "",
    timeout: float = 900,
) -> dict[str, Any]:
    """Generate one or more Seedream images through the Jwell relay."""
    import httpx

    Ark = _get_ark_class()
    request_headers = dict(headers)
    if idempotency_key and not any(
        key.lower() == "idempotency-key" for key in request_headers
    ):
        request_headers["Idempotency-Key"] = str(idempotency_key).strip()
    images = [str(url).strip() for url in image_urls if str(url).strip()]
    extra_body: dict[str, Any] = {}
    if str(provider or "").strip():
        extra_body["provider"] = str(provider).strip()
    if str(aspect_ratio or "").strip():
        extra_body["aspect_ratio"] = str(aspect_ratio).strip()
    if watermark is not None:
        extra_body["watermark"] = bool(watermark)
    if quantity is not None and int(quantity) > 1:
        extra_body["n"] = int(quantity)
    if watermark is not None:
        extra_body["watermark"] = bool(watermark)

    client = httpx.Client(headers=request_headers, timeout=timeout)
    try:
        with Ark(
            base_url=base_url.rstrip("/") + "/",
            api_key="internal-relay",
            timeout=timeout,
            max_retries=0,
            http_client=client,
        ) as ark:
            # Jwell returns its normalized `{object, data}` envelope, which is
            # intentionally smaller than Ark's strict ImagesResponse schema.
            # Keep Ark's transport/authentication, but parse the response as a
            # dictionary so a successful, billed generation is not rejected by
            # client-side schema validation.
            body: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "image": images or None,
                "response_format": "url",
                "size": _image_size(resolution),
            }
            body.update(extra_body)
            response = ark.post(
                "/images/generations",
                cast_to=dict[str, Any],
                body=body,
                options={"timeout": timeout},
            )
            return dict(response) if isinstance(response, Mapping) else {}
    finally:
        if not client.is_closed:
            client.close()
