"""Official Volcengine Ark SDK bridge for Seedance task requests.

The SDK is deliberately pointed at Jwell's internal relay instead of the
upstream Ark endpoint. Jwell remains responsible for model selection,
identity, preflight credits, settlement, and idempotency.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _get_ark_class() -> Any:
    """Import the preinstalled official Ark SDK."""
    try:
        from volcenginesdkarkruntime import Ark
    except ImportError as exc:
        raise RuntimeError(
            "The official Volcengine Ark SDK is unavailable; reinstall "
            "hermes-agent with its core dependencies."
        ) from exc
    return Ark


def _model_to_dict(value: Any) -> dict[str, Any]:
    """Convert an Ark response model without depending on its Pydantic version."""
    if hasattr(value, "to_dict"):
        converted = value.to_dict(use_api_names=True, exclude_unset=False)
    elif hasattr(value, "model_dump"):
        converted = value.model_dump(by_alias=True, exclude_unset=False)
    elif isinstance(value, Mapping):
        converted = dict(value)
    else:
        converted = {}
    return dict(converted) if isinstance(converted, Mapping) else {}


def _content_part(kind: str, url: str, role: str = "") -> dict[str, Any]:
    if kind == "image_url":
        part: dict[str, Any] = {"type": kind, "image_url": {"url": url}}
    else:
        part = {"type": kind, "audio_url": {"url": url}}
    if role:
        part["role"] = role
    return part


def _normalize_image_role(value: Any) -> str:
    """Map user-facing reference labels to the roles accepted by Seedance."""
    role = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "keyframe": "first_frame",
        "start_frame": "first_frame",
        "start": "first_frame",
        "first": "first_frame",
        "end_frame": "last_frame",
        "end": "last_frame",
        "last": "last_frame",
        "reference": "reference_image",
        "visual_reference": "reference_image",
    }
    role = aliases.get(role, role)
    return role if role in {"first_frame", "last_frame", "reference_image"} else ""


def _native_content(
    prompt: str,
    image_urls: Iterable[str],
    audio_urls: Iterable[str],
    image_roles: Iterable[str] = (),
    audio_roles: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build the native Ark content array used by Seedance v3."""
    content: list[dict[str, Any]] = []
    image_values = [str(url).strip() for url in image_urls if str(url).strip()]
    audio_values = [str(url).strip() for url in audio_urls if str(url).strip()]
    if prompt.strip():
        content.append({"type": "text", "text": prompt})

    image_role_values = list(image_roles)
    for index, url in enumerate(image_values):
        role = _normalize_image_role(image_role_values[index]) if index < len(image_role_values) else ""
        # Ark's native content schema requires a role for image inputs. A
        # single unlabelled image is the conventional image-to-video first
        # frame; additional unlabelled images are references rather than frame
        # boundaries. Explicit roles from the request remain authoritative.
        if not role:
            role = "first_frame" if len(image_values) == 1 else "reference_image"
        content.append(_content_part("image_url", url, role))

    audio_role_values = list(audio_roles)
    for index, url in enumerate(audio_values):
        role = (audio_role_values[index].strip() if index < len(audio_role_values) else "") or "soundtrack"
        content.append(_content_part("audio_url", url, role))
    return content


def create_seedance_task(
    *,
    base_url: str,
    headers: Mapping[str, str],
    model: str,
    prompt: str,
    image_urls: Iterable[str] = (),
    audio_urls: Iterable[str] = (),
    image_roles: Iterable[str] = (),
    audio_roles: Iterable[str] = (),
    ratio: str = "",
    resolution: str = "",
    duration: int | None = None,
    generate_audio: bool | None = None,
    timeout: float = 900,
) -> dict[str, Any]:
    """Create one task through Ark while sending it to Jwell's relay."""
    import httpx

    Ark = _get_ark_class()
    request_headers = dict(headers)
    content = _native_content(
        prompt,
        tuple(str(url).strip() for url in image_urls if str(url).strip()),
        tuple(str(url).strip() for url in audio_urls if str(url).strip()),
        image_roles,
        audio_roles,
    )
    if not content:
        raise ValueError("Seedance video request requires prompt or media content")

    client = httpx.Client(headers=request_headers, timeout=timeout)
    try:
        with Ark(
            base_url=base_url.rstrip("/") + "/",
            api_key="internal-relay",
            timeout=timeout,
            max_retries=0,
            http_client=client,
        ) as ark:
            response = ark.content_generation.tasks.create(
                model=model,
                content=content,
                ratio=ratio or None,
                resolution=resolution or None,
                duration=duration,
                generate_audio=generate_audio,
                timeout=timeout,
            )
            return _model_to_dict(response)
    finally:
        # Ark closes the client on normal exit; this also covers constructor
        # failures before the context manager is entered.
        if not client.is_closed:
            client.close()
