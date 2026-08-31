#!/usr/bin/env python3
"""Alphart app tool bridge.

Hermes owns the reasoning loop and creative orchestration. Alphart Edu exposes
boring internal APIs for relay, persistence, S3, billing, and canvas/session
metadata; this module turns agent decisions into those API calls without writing
temporary local artifacts.
"""

from __future__ import annotations

import contextlib
import contextvars
import ast
import base64
import binascii
import hashlib
import io
import json
import mimetypes
import os
import re
import uuid
import wave
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote, urlparse

import requests

from tools.registry import registry


_alphart_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "alphart_context", default={}
)


@contextlib.contextmanager
def alphart_context(values: Dict[str, Any]) -> Iterator[None]:
    token = _alphart_context.set(dict(values or {}))
    try:
        yield
    finally:
        _alphart_context.reset(token)


def _ctx() -> Dict[str, Any]:
    return _alphart_context.get() or {}


def _latest_canvas_created_node_id(item_type: str) -> str:
    """Return the newest node created during this Canvas agent turn."""
    wanted = str(item_type or "").strip().lower()
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas" or not wanted:
        return ""
    for node in reversed(_ctx().get("_canvas_created_nodes") or []):
        if not isinstance(node, dict):
            continue
        if str(node.get("item_type") or "").strip().lower() != wanted:
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id:
            return node_id
    return ""


def _canvas_context_video_item_id() -> str:
    """Return a trusted existing Canvas video target, if the request has one."""
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return ""
    item_id = str(_ctx().get("canvas_item_id") or "").strip()
    if not item_id:
        return ""
    item_type = str(_ctx().get("canvas_item_type") or "").strip().lower()
    if item_type == "video":
        return item_id
    if item_type:
        return ""
    # Older callers did not send canvas_item_type. Preserve their existing
    # video target unless the selected-node metadata explicitly identifies the
    # same id as a non-video node.
    selected_id = str(_ctx().get("selected_canvas_item_id") or "").strip()
    selected_type = str(_ctx().get("selected_canvas_item_type") or "").strip().lower()
    if selected_id == item_id and selected_type and selected_type != "video":
        return ""
    return item_id


def _tool_error(message: str, code: str = "") -> str:
    payload = {"success": False, "error": message}
    if code:
        payload["code"] = code
    return json.dumps(payload, ensure_ascii=False)


def _system_busy_tool_error() -> str:
    return "generate fail"


def _canvas_image_reference_key(value: Any) -> str:
    """Return only the Canvas S3 identity for a media reference.

    Canvas resolves these keys through its own storage service. A URL supplied
    by the model can be stale, provider-inaccessible, or point outside the
    Canvas document, so it must never be forwarded as a trusted keyframe.
    """
    if isinstance(value, dict):
        return str(
            value.get("s3_object_name")
            or value.get("object_key")
            or value.get("result_image_object_key")
            or ""
        ).strip()
    raw = str(value or "").strip()
    if raw.startswith(("http://", "https://")):
        return ""
    return raw


def _canvas_explicit_video_request() -> bool:
    """Keep speculative keyframe generation out of an explicit video turn."""
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return False
    text = str(_ctx().get("user_message") or _ctx().get("canvas_prompt_context") or "").strip()
    if not text:
        return False
    if re.search(r"\[skill:video-[^\]]+\]", text, flags=re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"(?:generate|create|make|render|produce|animate|生成|创建|製作|制作|渲染)"
            r"[^.!?。！？]{0,80}(?:video|clip|animation|视频|动画|短片)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _video_duration_seconds_from_text(text: Any) -> int:
    """Extract an explicit video duration from the user's request."""
    value = str(text or "")
    for pattern in (
        r"\b(\d{1,3})\s*(?:seconds?|secs?|s)\b",
        r"\b(\d{1,3})\s*秒",
    ):
        match = re.search(pattern, value, re.IGNORECASE)
        if not match:
            continue
        try:
            seconds = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return 0


def _positive_video_duration(value: Any) -> Optional[int]:
    """Return a valid whole-second duration, treating zero as omitted."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _strip_audio_preferences(text: str) -> str:
    return re.sub(r"\n*\s*<audio_preferences\b[^>]*/>\s*", "\n", str(text or ""), flags=re.I).strip()


def _audio_language_type_from_text(text: Any) -> str:
    plain_text = _strip_audio_preferences(str(text or ""))
    value = plain_text.lower()
    if any(word in value for word in ("粤语", "粵語", "广东话", "廣東話", "cantonese", "yue")):
        return "cantonese"
    if any(word in value for word in ("english", "英文", "英语", "英語")):
        return "english"
    preference = re.search(r"<audio_preferences\b[^>]*\blanguage_type=[\"']([^\"']+)[\"']", str(text or ""), re.I)
    if preference:
        preferred = preference.group(1).strip().lower()
        if preferred in {"mandarin", "cantonese", "english"}:
            return preferred
    if re.search(r"[\u4e00-\u9fff]", plain_text):
        return "mandarin"
    return "english"


def _clean_audio_topic(text: Any) -> str:
    value = _strip_audio_preferences(str(text or ""))
    value = re.sub(
        r"^\s*(/audio|generate\s+(an?\s+)?audio|create\s+(an?\s+)?audio|generate\s+speech|create\s+speech|"
        r"生成一段?音频|生成一段?音訊|生成音频|生成音訊|生成语音|生成語音|生成旁白)\s*[:：,，-]*\s*",
        "",
        value,
        flags=re.I,
    ).strip()
    value = re.sub(r"\b(use|in|with)\s+(mandarin|cantonese|english)\b", "", value, flags=re.I).strip()
    value = re.sub(r"(用|以)?(中文|普通话|普通話|粤语|粵語|广东话|廣東話|英文|英语|英語)(介绍|介紹|朗读|朗讀|讲解|講解)?", "", value).strip()
    return value or _strip_audio_preferences(str(text or "")).strip()


def _audio_script_from_request(text: Any, language_type: str = "") -> str:
    topic = _clean_audio_topic(text)
    language = (language_type or _audio_language_type_from_text(text)).strip().lower()
    if language == "cantonese":
        return (
            f"大家好，今日我哋用一段簡單清楚嘅講解，認識{topic}。\n\n"
            f"首先，我哋會由最基本嘅概念開始，了解{topic}係乜嘢，點解佢重要。"
            "然後，我哋會用生活入面容易見到嘅例子，將抽象嘅內容變得更加具體。"
            "聽嘅時候，可以留意三個重點：第一，事情點樣開始；第二，中間經過咩變化；第三，最後會產生咩結果。\n\n"
            f"總結嚟講，{topic}唔係孤立嘅知識點，而係一個有因有果、可以一步一步理解嘅過程。"
            "只要抓住主要概念，再配合例子，就會更容易記住同應用。"
        )
    if language == "english":
        return (
            f"Hello. In this short audio lesson, we will explain {topic} in a clear and learner-friendly way.\n\n"
            f"First, we will start with the basic idea: what {topic} means and why it matters. "
            "Then we will connect the idea to an everyday example, so the concept becomes easier to picture. "
            "As you listen, focus on three things: what starts the process, what changes during the process, "
            "and what result comes at the end.\n\n"
            f"In summary, {topic} is easier to understand when we break it into simple steps. "
            "Once the key idea is clear, examples can help you remember it and use it in new situations."
        )
    return (
        f"大家好，下面用一段简洁清楚的音频，来介绍{topic}。\n\n"
        f"首先，我们从最基本的概念开始，理解{topic}是什么，以及它为什么重要。"
        "接着，我们会结合生活中容易观察到的例子，把抽象的内容变得更具体。"
        "在听的过程中，可以重点关注三个问题：第一，它是怎样开始的；第二，中间发生了什么变化；第三，最后产生了什么结果。\n\n"
        f"总结一下，{topic}并不是孤立的知识点，而是一个可以分步骤理解的过程。"
        "只要抓住核心概念，再配合具体例子，就能更容易记住，并在学习和表达中灵活使用。"
    )


def _infer_storybook_language(text: Any) -> str:
    value = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", value):
        if re.search(r"[繪書頁學習兒童臺灣繁體]", value):
            return "zh-TW"
        return "zh-CN"
    return "en"


def _explicit_storybook_cantonese_read_aloud(text: Any) -> bool:
    value = str(text or "").lower()
    has_cantonese = any(word in value for word in ("粤语", "粵語", "广东话", "廣東話", "cantonese", "yue"))
    has_read_aloud = any(
        word in value
        for word in (
            "朗读",
            "朗讀",
            "读",
            "讀",
            "配音",
            "旁白",
            "audio",
            "speech",
            "voice",
            "voiceover",
            "read aloud",
            "narrate",
            "tts",
        )
    )
    return has_cantonese and has_read_aloud


def _normalize_audio_language_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"cantonese", "yue", "zh-hk", "zh_hk"} or any(
        word in normalized for word in ("粤语", "粵語", "广东话", "廣東話")
    ):
        return "cantonese"
    if normalized in {"mandarin", "zh", "zh-cn", "zh_cn", "zh-tw", "zh_tw", "chinese"} or any(
        word in normalized for word in ("中文", "普通话", "普通話")
    ):
        return "mandarin"
    if normalized in {"english", "en"} or any(word in normalized for word in ("english", "英文", "英语", "英語")):
        return "english"
    return ""


def _storybook_read_aloud_language(text: Any, language: Any = "", requested: Any = "") -> str:
    if _explicit_storybook_cantonese_read_aloud(text):
        return "cantonese"
    normalized_requested = _normalize_audio_language_type(requested)
    if normalized_requested:
        return normalized_requested
    lang = str(language or "").lower()
    if any(token in lang for token in ("zh", "chinese", "中文")) or re.search(r"[\u4e00-\u9fff]", str(text or "")):
        return "mandarin"
    return "english"


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "model"


def _selected_tools() -> Iterable[Dict[str, Any]]:
    raw = _ctx().get("tool_list") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _pick_tool(media_type: str, args: Dict[str, Any]) -> Dict[str, Any]:
    tool_id = str(args.get("tool_id") or args.get("id") or "").strip()
    provider = str(args.get("provider") or args.get(f"{media_type}_provider") or "").strip()
    model = str(args.get("model") or args.get(f"{media_type}_model") or "").strip()
    candidates = []
    for tool in _selected_tools():
        tool_type = str(tool.get("type") or tool.get("model_type") or "").lower()
        tool_provider = str(tool.get("provider") or "").strip()
        tool_model = str(tool.get("model") or tool.get("name") or tool.get("key") or "").strip()
        if tool_type and tool_type != media_type:
            continue
        if tool_id and str(tool.get("id") or "") != tool_id:
            continue
        if provider and tool_provider and tool_provider != provider:
            continue
        if model and tool_model and tool_model != model:
            continue
        candidates.append(tool)
    if candidates:
        return candidates[0]
    return {
        "id": f"generate_{media_type}_by_{_slug(provider)}_{_slug(model)}",
        "type": media_type,
        "provider": provider,
        "model": model,
    }


def _tool_model_value(tool: Dict[str, Any]) -> str:
    return str(tool.get("model") or tool.get("name") or tool.get("key") or "").strip()


def _set_tool_defaults(args: Dict[str, Any], tool: Dict[str, Any]) -> None:
    provider = str(tool.get("provider") or "").strip()
    model = _tool_model_value(tool)
    if provider and not args.get("provider"):
        args["provider"] = provider
    if model and not args.get("model"):
        args["model"] = model


def _set_canvas_model_default(args: Dict[str, Any], media_type: str) -> None:
    """Use the frontend-selected provider:model when the model omits tool args."""
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return
    selected = str(_ctx().get(f"{media_type}_model") or "").strip()
    if not selected:
        return
    provider, separator, model = selected.partition(":")
    if separator and provider and model:
        args.setdefault("provider", provider)
        args.setdefault("model", model)
    else:
        args.setdefault("model", selected)


def _explicit_audio_preference(text: Any) -> Optional[bool]:
    """Read an explicit audio on/off instruction without changing Edu defaults."""
    value = str(text or "")
    no_audio = re.search(
        r"(?:without|no|mute|silent|disable(?:d)?|不要|无|無|没有|沒有|不含|不带|不帶)\s*(?:any\s+)?"
        r"(?:audio|sound|music|sound\s+track|soundtrack|bgm|background\s+music|声音|聲音|音频|音訊|配乐|背景音乐)",
        value,
        flags=re.IGNORECASE,
    )
    if no_audio:
        return False
    with_audio = re.search(
        r"(?:with|include|including|enable(?:d)?|要|有)\s*(?:(?:an?|the)\s+)?"
        r"(?:audio|sound|music|sound\s+track|soundtrack|bgm|background\s+music|声音|聲音|音频|音訊|配乐|背景音乐)",
        value,
        flags=re.IGNORECASE,
    )
    if with_audio:
        return True
    return None


def _log_model_value(value: Any) -> str:
    text = str(value or "").strip()
    return text or "backend-selected"


def _backend_url() -> str:
    context_url = str(
        _ctx().get("application_backend_url")
        or _ctx().get("backend_url")
        or ""
    ).strip()
    app_scope = str(_ctx().get("app_scope") or "edu").strip().lower()
    if app_scope == "canvas":
        value = str(os.getenv("ALPHART_CANVAS_BACKEND_URL") or os.getenv("CANVAS_BACKEND_URL") or context_url or "").strip()
    else:
        value = str(os.getenv("ALPHART_EDU_BACKEND_URL") or context_url or "").strip()
    return value.rstrip("/")


def _auth_token() -> str:
    return str(
        _ctx().get("auth_token")
        or os.getenv("ALPHART_AUTH_TOKEN")
        or os.getenv("CANVAS_AUTH_TOKEN")
        or ""
    ).strip()


def _service_token() -> str:
    return str(
        os.getenv("HERMES_AGENT_TOKEN")
        or os.getenv("ALPHART_AGENT_TOKEN")
        or os.getenv("CANVAS_AGENT_TOKEN")
        or ""
    ).strip()


def _internal_relay_url(path: str) -> str:
    normalized_path = path.lstrip("/")
    backend_url = _backend_url()
    if not backend_url:
        return ""
    return f"{backend_url}/internal/v1/{normalized_path}"


def _jwell_relay_enabled() -> bool:
    return (
        str(_ctx().get("app_scope") or "edu").strip().lower() != "canvas"
        and bool(_jwell_relay_base_url())
        and bool(_jwell_relay_app_secret())
    )


def _jwell_relay_base_url() -> str:
    raw = str(os.getenv("JWELL_SERVICE_GRPC_ADDRS") or os.getenv("JWELL_SERVICE_GRPC_ADDR") or "").strip()
    address = raw.split(",", 1)[0].strip().rstrip("/")
    if not address:
        return ""
    if address.startswith(("http://", "https://")):
        return address
    return "http://" + address


def _jwell_relay_app_secret() -> str:
    return str(os.getenv("JWELL_APP_SECRET") or "").strip()


def _is_seedance_video_model(provider: Any, model: Any) -> bool:
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip().lower()
    return provider == "seedance" or (
        provider == "byteplus"
        and (
            model == "seedance"
            or model.startswith("seedance-")
            or model.startswith("dreamina-seedance-")
            or model.startswith("doubao-seedance-")
            or model.startswith("bytedance-seedance-")
        )
    )


def _is_seedream_image_model(provider: Any, model: Any) -> bool:
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip().lower()
    return provider == "seedream" or (
        provider in {"byteplus", "volcengine", "doubao"} and "seedream" in model
    )


def _relay_url(path: str, *, provider: Any = "", model: Any = "") -> str:
    if _jwell_relay_enabled():
        if path.strip("/").lower() == "videos" and _is_seedance_video_model(provider, model):
            return f"{_jwell_relay_base_url()}/internal/v3/contents/generations/tasks"
        return f"{_jwell_relay_base_url()}/internal/v1/{path.lstrip('/')}"
    return _internal_relay_url(path)


def _internal_api_url(path: str) -> str:
    backend_url = _backend_url()
    if not backend_url:
        return ""
    return f"{backend_url}/internal/api/v1/{path.lstrip('/')}"


def _backend_tool_timeout(default: int = 900) -> int:
    # Keep the shared timeout opt-in explicit. A stale Edu-only value must not
    # shorten Canvas media relays, while Edu continues to honor its legacy
    # setting until the shared variable is adopted everywhere.
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        raw = (
            os.getenv("ALPHART_BACKEND_TOOL_TIMEOUT_SECONDS")
            or os.getenv("CANVAS_BACKEND_TOOL_TIMEOUT_SECONDS")
            or str(default)
        )
    else:
        raw = (
            os.getenv("ALPHART_EDU_BACKEND_TOOL_TIMEOUT_SECONDS")
            or os.getenv("ALPHART_BACKEND_TOOL_TIMEOUT_SECONDS")
            or str(default)
        )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _game_browserless_url() -> str:
    return str(
        os.getenv("ALPHART_EDU_BROWSERLESS_URL")
        or os.getenv("BROWSERLESS_URL")
        or ""
    ).strip().rstrip("/")


def _game_browserless_token() -> str:
    return str(
        os.getenv("ALPHART_EDU_BROWSERLESS_TOKEN")
        or os.getenv("BROWSERLESS_TOKEN")
        or ""
    ).strip()


def _internal_relay_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Authorization": "Bearer internal-relay"}
    service_token = _service_token()
    if service_token:
        headers["X-Hermes-Agent-Token"] = service_token
    if _ctx().get("user_id"):
        headers["X-Internal-User-ID"] = str(_ctx().get("user_id"))
    if _ctx().get("user_uuid"):
        headers["X-Internal-User-UUID"] = str(_ctx().get("user_uuid"))
    org_no = str(_ctx().get("org_no") or _ctx().get("storage_prefix") or "").strip()
    if org_no:
        headers["X-Org-No"] = org_no
    if _ctx().get("session_id"):
        headers["X-Session-ID"] = str(_ctx().get("session_id"))
    if _ctx().get("canvas_id"):
        headers["X-Canvas-ID"] = str(_ctx().get("canvas_id"))
    return headers


def _relay_headers(idempotency_key: Any = "") -> Dict[str, str]:
    headers = _internal_relay_headers()
    if _jwell_relay_enabled():
        headers["X-App-Secret"] = _jwell_relay_app_secret()
    key = str(idempotency_key or "").strip()
    if key:
        headers["Idempotency-Key"] = key
    return headers


def _seedance_media_roles(references: Any) -> list[str]:
    if not isinstance(references, list):
        references = [references] if references else []
    roles: list[str] = []
    for reference in references:
        if isinstance(reference, dict):
            roles.append(str(reference.get("role") or "").strip())
        else:
            roles.append("")
    return roles


def _resolve_jwell_media_urls(media_type: str, references: Any) -> list[str]:
    if not _jwell_relay_enabled() or not references:
        return list(references or [])
    if not isinstance(references, list):
        references = [references]
    url = _internal_api_url("agent/relay-media-urls")
    if not url:
        raise RuntimeError("ALPHART_EDU_BACKEND_URL is not configured")
    response = requests.post(
        url,
        json={"media_type": media_type, "references": references},
        headers=_internal_relay_headers(),
        timeout=_backend_tool_timeout(),
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list) or not all(isinstance(item, str) and item for item in data):
        raise RuntimeError("Edu media bridge returned invalid provider URLs")
    return data


def _import_jwell_media(asset: Dict[str, Any], media_type: str, object_name: str = "") -> Dict[str, Any]:
    if not _jwell_relay_enabled():
        return asset
    media_url = str(asset.get("url") or asset.get(f"{media_type}_url") or "").strip()
    if not media_url:
        raise RuntimeError("Jwell relay returned no media URL")
    url = _internal_api_url("agent/relay-media-import")
    if not url:
        raise RuntimeError("ALPHART_EDU_BACKEND_URL is not configured")
    request_payload = {
        "url": media_url,
        "mime_type": asset.get("mime_type") or "",
        "media_type": media_type,
        "canvas_id": _ctx().get("canvas_id") or "",
    }
    if str(object_name or "").strip():
        request_payload["object_name"] = str(object_name).strip()
    response = requests.post(
        url,
        json=request_payload,
        headers=_internal_relay_headers(),
        timeout=_backend_tool_timeout(),
    )
    response.raise_for_status()
    stored = response.json().get("data")
    if not isinstance(stored, dict) or not str(stored.get("s3_object_name") or "").strip():
        raise RuntimeError("Edu media bridge returned no stored asset")
    merged = dict(asset)
    merged.update(stored)
    # The bridge returns the persistent URL as `url`, while the provider
    # response may still carry the original data URI under its media alias.
    # Keep every alias pointed at the stored object so a large base64 payload
    # cannot escape in the agent response after a successful import.
    stored_url = str(stored.get("url") or "").strip()
    if not stored_url or stored_url.startswith("data:"):
        raise RuntimeError("Edu media bridge returned no persistent media URL")
    merged["url"] = stored_url
    merged[f"{media_type}_url"] = stored_url
    merged["provider"] = asset.get("provider") or merged.get("provider")
    merged["model"] = asset.get("model") or merged.get("model")
    merged["_credit_settled"] = True
    return merged


_AUDIO_CHUNK_ENGLISH_WORDS = 75
_AUDIO_CHUNK_CJK_CHARS = 180
_AUDIO_CHUNK_PAUSE_SECONDS = 0.35


def _audio_has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text or ""))


def _audio_text_units(text: str) -> int:
    cjk_count = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text or ""))
    non_cjk = re.sub(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text or "")
    word_count = len(re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", non_cjk, flags=re.UNICODE))
    return cjk_count + word_count


def _audio_chunk_limit(text: str, language_type: str) -> int:
    language = str(language_type or "").strip().lower()
    if language in {"cantonese", "mandarin", "chinese", "zh", "yue"} or _audio_has_cjk(text):
        return _AUDIO_CHUNK_CJK_CHARS
    return _AUDIO_CHUNK_ENGLISH_WORDS


def _split_audio_long_segment(segment: str, limit: int) -> List[str]:
    segment = str(segment or "").strip()
    if not segment:
        return []
    if _audio_text_units(segment) <= limit:
        return [segment]
    if _audio_has_cjk(segment):
        runes = list(segment)
        return ["".join(runes[index:index + limit]).strip() for index in range(0, len(runes), limit)]
    words = segment.split()
    chunks: List[str] = []
    current: List[str] = []
    units = 0
    for word in words:
        word_units = _audio_text_units(word)
        if current and units + word_units > limit:
            chunks.append(" ".join(current).strip())
            current = []
            units = 0
        current.append(word)
        units += word_units
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _split_audio_script(text: str, language_type: str = "") -> List[str]:
    """Split spoken text at natural punctuation into approximately 30s chunks."""
    text = str(text or "").strip()
    if not text:
        return []
    limit = _audio_chunk_limit(text, language_type)
    segments: List[str] = []
    current: List[str] = []
    characters = list(text)
    for index, character in enumerate(characters):
        current.append(character)
        next_character = characters[index + 1] if index + 1 < len(characters) else ""
        is_decimal = character == "." and current and next_character.isdigit() and current[-2:-1] and current[-2].isdigit()
        if character in "。！？；!?;" or (character == "." and not is_decimal) or character == "\n":
            if character == "\n" and next_character == "\n":
                segments.append("".join(current).strip())
                current = []
            elif character != "\n":
                segments.append("".join(current).strip())
                current = []
    if current:
        segments.append("".join(current).strip())

    chunks: List[str] = []
    pending: List[str] = []
    pending_units = 0
    for segment in segments:
        if not segment:
            continue
        for part in _split_audio_long_segment(segment, limit):
            part_units = _audio_text_units(part)
            if pending and pending_units + part_units > limit:
                chunks.append(" ".join(pending).strip())
                pending = []
                pending_units = 0
            pending.append(part)
            pending_units += part_units
    if pending:
        chunks.append(" ".join(pending).strip())
    return [chunk for chunk in chunks if chunk]


def _audio_asset_bytes(asset: Dict[str, Any]) -> bytes:
    media_url = str(asset.get("url") or asset.get("audio_url") or "").strip()
    if not media_url:
        raise RuntimeError("audio chunk returned no playable URL")
    if media_url.startswith("data:"):
        header, encoded = media_url.split(",", 1)
        if ";base64" not in header.lower():
            raise RuntimeError("audio chunk data URL is not base64 encoded")
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(f"audio chunk data URL is invalid: {exc}") from exc
    response = requests.get(media_url, timeout=_backend_tool_timeout())
    response.raise_for_status()
    return response.content


def _concat_audio_wav_chunks(chunks: List[bytes]) -> Tuple[bytes, float]:
    if not chunks:
        raise RuntimeError("no audio chunks to concatenate")
    if len(chunks) == 1:
        with wave.open(io.BytesIO(chunks[0]), "rb") as source:
            return chunks[0], source.getnframes() / max(source.getframerate(), 1)

    output = io.BytesIO()
    output_wave = None
    duration = 0.0
    params = None
    try:
        for index, chunk in enumerate(chunks):
            with wave.open(io.BytesIO(chunk), "rb") as source:
                current_params = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                if params is None:
                    params = current_params
                    output_wave = wave.open(output, "wb")
                    output_wave.setnchannels(params[0])
                    output_wave.setsampwidth(params[1])
                    output_wave.setframerate(params[2])
                    output_wave.setcomptype("NONE", "not compressed")
                elif current_params != params:
                    raise RuntimeError("audio chunks use incompatible WAV formats")
                if index > 0:
                    silence_frames = int(params[2] * _AUDIO_CHUNK_PAUSE_SECONDS)
                    output_wave.writeframes(b"\x00" * silence_frames * params[0] * params[1])
                    duration += _AUDIO_CHUNK_PAUSE_SECONDS
                frames = source.getnframes()
                output_wave.writeframes(source.readframes(frames))
                duration += frames / max(params[2], 1)
    finally:
        if output_wave is not None:
            output_wave.close()
    return output.getvalue(), duration


def _audio_result_payload(result: str) -> Dict[str, Any]:
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return {}
    if not isinstance(decoded, dict) or decoded.get("success") is False:
        return {}
    payload = decoded.get("result")
    return payload if isinstance(payload, dict) else decoded


def _audio_result_success(result: str) -> bool:
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return False
    if not isinstance(decoded, dict) or decoded.get("success") is False:
        return False
    if str(decoded.get("status") or "").strip().lower() in {"failed", "error", "failure"}:
        return False
    payload = decoded.get("result")
    if isinstance(payload, dict) and str(payload.get("status") or "").strip().lower() in {"failed", "error", "failure", "partial"}:
        return False
    return bool(_audio_result_payload(result))


def _audio_usage_sum(total: Dict[str, Any], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def _stable_audio_object_name(base_call_id: str) -> str:
    digest = hashlib.sha256(str(base_call_id or "").strip().encode("utf-8")).hexdigest()
    return f"audio-{digest}.wav"


def _generate_chunked_audio(
    args: Dict[str, Any],
    text: str,
    chunks: List[str],
    tool_call_id: str = "",
) -> str:
    base_call_id = str(tool_call_id or args.get("tool_call_id") or uuid.uuid4()).strip()
    generated_assets: List[Dict[str, Any]] = []
    audio_chunks: List[bytes] = []
    total_usage: Dict[str, Any] = {}
    provider = str(args.get("provider") or "").strip()
    model = str(args.get("model") or "").strip()
    retries = max(1, int(os.getenv("ALPHART_AUDIO_RELAY_RETRY_ATTEMPTS", "3") or "3"))
    for index, chunk in enumerate(chunks, start=1):
        chunk_args = dict(args)
        chunk_args["input"] = chunk
        chunk_args["tool_call_id"] = f"{base_call_id}:chunk:{index}"
        chunk_args["response_format"] = "wav"
        chunk_args["_audio_chunk"] = True
        chunk_args["_skip_media_import"] = True
        result = ""
        for attempt in range(1, retries + 1):
            result = _handle_alphart_generate_audio(chunk_args)
            if _audio_result_success(result):
                break
            print(
                f"[alphart-agent] audio chunk failed chunk={index}/{len(chunks)} "
                f"attempt={attempt}/{retries}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(2 * attempt, 5))
        if not _audio_result_success(result):
            return json.dumps({
                "status": "failed",
                "result": {
                    "type": "generate_audio_result",
                    "status": "partial",
                    "message": "generate fail",
                    "provider": provider,
                    "model": model,
                    "input": text,
                    "script": text,
                    "chunk_count": len(chunks),
                    "generated_chunk_count": len(audio_chunks),
                    "failed_chunk_index": index,
                    "usage": total_usage,
                },
            }, ensure_ascii=False)
        asset = _audio_result_payload(result)
        _audio_usage_sum(total_usage, asset.get("usage"))
        try:
            audio_chunks.append(_audio_asset_bytes(asset))
        except (requests.RequestException, RuntimeError) as exc:
            return json.dumps({
                "status": "failed",
                "result": {
                    "type": "generate_audio_result",
                    "status": "partial",
                    "message": str(exc),
                    "provider": provider,
                    "model": model,
                    "input": text,
                    "script": text,
                    "chunk_count": len(chunks),
                    "generated_chunk_count": len(audio_chunks),
                    "failed_chunk_index": index,
                    "usage": total_usage,
                },
            }, ensure_ascii=False)
        generated_assets.append(asset)
    try:
        combined, duration = _concat_audio_wav_chunks(audio_chunks)
    except (RuntimeError, wave.Error) as exc:
        return _tool_error(f"Unable to concatenate audio chunks: {exc}")
    combined_asset: Dict[str, Any] = {
        "url": "data:audio/wav;base64," + base64.b64encode(combined).decode("ascii"),
        "audio_url": "data:audio/wav;base64," + base64.b64encode(combined).decode("ascii"),
        "mime_type": "audio/wav",
        "provider": provider,
        "model": model,
    }
    if _jwell_relay_enabled():
        object_name = _stable_audio_object_name(base_call_id)
        import_attempts = 3
        last_import_error: Optional[BaseException] = None
        for attempt in range(1, import_attempts + 1):
            try:
                combined_asset = _import_jwell_media(combined_asset, "audio", object_name=object_name)
                last_import_error = None
                break
            except (requests.RequestException, RuntimeError) as exc:
                last_import_error = exc
                if attempt < import_attempts:
                    time.sleep(min(2 * attempt, 5))
        if last_import_error is not None:
            return json.dumps({
                "status": "failed",
                "result": {
                    "type": "generate_audio_result",
                    "status": "partial",
                    "message": f"Unable to store generated audio: {last_import_error}",
                    "provider": provider,
                    "model": model,
                    "input": text,
                    "script": text,
                    "chunk_count": len(chunks),
                    "generated_chunk_count": len(generated_assets),
                    "failed_stage": "media_import",
                    "usage": total_usage,
                },
            }, ensure_ascii=False)
    result = {
        "type": "generate_audio_result",
        "status": "completed",
        "provider": combined_asset.get("provider") or provider,
        "model": combined_asset.get("model") or model,
        "input": text,
        "script": text,
        "url": combined_asset.get("url"),
        "audio_url": combined_asset.get("url") or combined_asset.get("audio_url"),
        "mime_type": combined_asset.get("mime_type") or "audio/wav",
        "duration_seconds": duration,
        "filename": combined_asset.get("filename"),
        "s3_object_name": combined_asset.get("s3_object_name"),
        "chunk_count": len(chunks),
        "generated_chunk_count": len(generated_assets),
        "usage": total_usage,
        "_credit_settled": True,
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _handle_write_plan(args: Dict[str, Any], **_: Any) -> str:
    steps = []
    for item in (args or {}).get("steps") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if title:
            steps.append({"title": title, "description": description})
    lines = []
    for index, step in enumerate(steps, start=1):
        line = f"{index}. {step['title']}"
        if step.get("description"):
            line += f" - {step['description']}"
        lines.append(line)
    return "\n".join(lines)


def _require_canvas_graph_scope() -> str:
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return _tool_error("Canvas graph tools are only available in the Canvas app scope")
    return ""


def _require_canvas_mutation_allowed() -> str:
    if (
        str(_ctx().get("app_scope") or "").strip().lower() == "canvas"
        and _ctx().get("canvas_read_only_turn")
    ):
        return _tool_error(
            "This Canvas analysis turn is read-only; do not create, update, connect, or generate nodes.",
            "CANVAS_READ_ONLY",
        )
    return ""


def _handle_canvas_create_node(args: Dict[str, Any], **_: Any) -> str:
    scope_error = _require_canvas_graph_scope()
    if scope_error:
        return scope_error
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    args = dict(args or {})
    suppress_context_connections = bool(args.pop("_suppress_canvas_reference_connections", False))
    if not args.get("canvas_id"):
        args["canvas_id"] = _ctx().get("canvas_id")
    if not args.get("item_type") and args.get("type"):
        args["item_type"] = args.get("type")
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        item_type = str(args.get("item_type") or "text").strip().lower()
        source_ids = []
        for source_field in ("source_item_ids", "connect_from_item_ids", "reference_item_ids"):
            raw_sources = args.get(source_field) or []
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            source_ids.extend(str(value).strip() for value in raw_sources if str(value).strip())
        if not suppress_context_connections:
            context_sources = _ctx().get("reference_item_ids") or []
            if isinstance(context_sources, str):
                context_sources = [context_sources]
            source_ids.extend(str(value).strip() for value in context_sources if str(value).strip())
        # A new media node follows the newly created Prompt node. Persist both
        # the user's referenced inputs and that prompt-to-output edge in the
        # backend, even if the model forgets a separate connect tool call.
        if item_type in {"image", "video", "audio"}:
            for created in reversed(_ctx().get("_canvas_created_nodes") or []):
                if created.get("item_type") in {"text", "note"}:
                    prompt_id = str(created.get("id") or "").strip()
                    if prompt_id and prompt_id not in source_ids:
                        source_ids.append(prompt_id)
                    break
        if source_ids:
            args["source_item_ids"] = list(dict.fromkeys(source_ids))
    if not args.get("content"):
        content: Dict[str, Any] = {}
        if args.get("text"):
            content["text"] = args.get("text")
        if args.get("prompt"):
            content["prompt"] = args.get("prompt")
        if content:
            args["content"] = content
    url = _internal_api_url("canvas/nodes")
    if not url:
        return _tool_error("Canvas backend URL is not configured")
    try:
        resp = requests.post(
            url,
            json=args,
            headers=_internal_relay_headers(),
            timeout=_backend_tool_timeout(60),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Canvas node create failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    if resp.status_code < 200 or resp.status_code >= 300:
        detail = decoded.get("detail") if isinstance(decoded, dict) else resp.text
        return _tool_error(str(detail or f"HTTP {resp.status_code}"))
    if isinstance(decoded, dict):
        item_id = str(decoded.get("canvas_item_id") or (decoded.get("item") or {}).get("id") or "").strip()
        if item_id:
            _ctx().setdefault("_canvas_created_nodes", []).append({"id": item_id, "item_type": str(args.get("item_type") or "text").strip().lower()})
    return json.dumps({"status": "success", "result": decoded}, ensure_ascii=False)


def _ensure_canvas_image_generation_graph(prompt: str) -> Tuple[str, str]:
    """Materialize the new-image graph before the relay is called.

    Canvas requests without an explicit media target must leave behind both the
    enriched prompt and the image output node. This is also the recovery path
    when the model returns a prose conclusion instead of executing the graph
    tools. The guard keeps Edu's legacy image flow unchanged.
    """
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return "", ""
    if str(_ctx().get("canvas_item_id") or "").strip():
        return "", ""

    _ctx().setdefault("_canvas_created_nodes", [])
    prompt_node_id = _latest_canvas_created_node_id("text") or _latest_canvas_created_node_id("note")
    output_node_id = _latest_canvas_created_node_id("image")
    prompt_created = False

    if not prompt_node_id:
        source_ids = _ctx().get("reference_item_ids") or []
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        result = _handle_canvas_create_node({
            "canvas_id": _ctx().get("canvas_id"),
            "item_type": "text",
            "title": "Prompt",
            "text": prompt,
            "source_item_ids": list(dict.fromkeys(str(value).strip() for value in source_ids if str(value).strip())),
        })
        if not _canvas_tool_succeeded(result):
            return "", "Canvas prompt node creation failed"
        prompt_node_id = _latest_canvas_created_node_id("text") or _latest_canvas_created_node_id("note")
        prompt_created = bool(prompt_node_id)

    if not output_node_id:
        result = _handle_canvas_create_node({
            "canvas_id": _ctx().get("canvas_id"),
            "item_type": "image",
            "title": "Image",
            "prompt": prompt,
        })
        if not _canvas_tool_succeeded(result):
            return "", "Canvas image node creation failed"
        output_node_id = _latest_canvas_created_node_id("image")
    elif prompt_created:
        result = _handle_canvas_connect_nodes({
            "canvas_id": _ctx().get("canvas_id"),
            "source_item_id": prompt_node_id,
            "target_item_id": output_node_id,
        })
        if not _canvas_tool_succeeded(result):
            return "", "Canvas prompt-to-image connection failed"

    if not prompt_node_id or not output_node_id:
        return "", "Canvas image graph was not created"
    return output_node_id, ""


def _ensure_canvas_video_generation_graph(prompt: str) -> Tuple[str, str]:
    """Materialize a new Canvas prompt/video graph before video relay.

    The video relay updates an existing video node, so a homepage request (or a
    model tool call that omitted the newly-created id) must create the target
    node before submitting the provider task. This is Canvas-only; Edu keeps its
    existing relay contract.
    """
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return "", ""
    if _canvas_context_video_item_id():
        return "", ""
    if not str(_ctx().get("canvas_id") or "").strip():
        return "", "Canvas canvas_id is required before creating a video node"

    _ctx().setdefault("_canvas_created_nodes", [])
    # Reuse nodes the model created earlier in this same turn. Once a relay has
    # been submitted, a second automatic request gets a fresh pair instead of
    # overwriting the first generation.
    prompt_node_id = ""
    output_node_id = ""
    source_ids = _ctx().get("reference_item_ids") or []
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_ids = [str(value).strip() for value in source_ids if str(value).strip()]
    selected_id = str(_ctx().get("selected_canvas_item_id") or "").strip()
    selected_type = str(_ctx().get("selected_canvas_item_type") or "").strip().lower()
    if selected_id and selected_type in {"text", "note", "image", "audio"}:
        source_ids.append(selected_id)
    source_ids = list(dict.fromkeys(source_ids))

    if not _ctx().get("_canvas_generation_submitted"):
        prompt_node_id = _latest_canvas_created_node_id("text") or _latest_canvas_created_node_id("note")
        output_node_id = _latest_canvas_created_node_id("video")

    prompt_created = False
    if not prompt_node_id:
        result = _handle_canvas_create_node({
            "canvas_id": _ctx().get("canvas_id"),
            "item_type": "text",
            "title": "Prompt",
            "text": prompt,
            "_suppress_canvas_reference_connections": True,
        })
        if not _canvas_tool_succeeded(result):
            return "", "Canvas prompt node creation failed"
        prompt_node_id = _latest_canvas_created_node_id("text") or _latest_canvas_created_node_id("note")
        prompt_created = bool(prompt_node_id)

    if not output_node_id:
        result = _handle_canvas_create_node({
            "canvas_id": _ctx().get("canvas_id"),
            "item_type": "video",
            "title": "Video",
            "prompt": prompt,
            "source_item_ids": source_ids,
        })
        if not _canvas_tool_succeeded(result):
            return "", "Canvas video node creation failed"
        output_node_id = _latest_canvas_created_node_id("video")
    else:
        for src_id in source_ids:
            if src_id and src_id != output_node_id:
                conn_result = _handle_canvas_connect_nodes({
                    "canvas_id": _ctx().get("canvas_id"),
                    "source_item_id": src_id,
                    "target_item_id": output_node_id,
                })
                if not _canvas_tool_succeeded(conn_result):
                    return "", "Canvas source connection failed"
        if prompt_created:
            result = _handle_canvas_connect_nodes({
                "canvas_id": _ctx().get("canvas_id"),
                "source_item_id": prompt_node_id,
                "target_item_id": output_node_id,
            })
            if not _canvas_tool_succeeded(result):
                return "", "Canvas prompt-to-video connection failed"
    if not output_node_id:
        return "", "Canvas video graph was not created"
    return output_node_id, ""


def _canvas_tool_succeeded(result: str) -> bool:
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(decoded, dict)
        and decoded.get("success") is not False
        and decoded.get("status") not in {"failed", "error", "failure"}
    )


def _handle_canvas_update_node(args: Dict[str, Any], **_: Any) -> str:
    scope_error = _require_canvas_graph_scope()
    if scope_error:
        return scope_error
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    args = dict(args or {})
    item_id = str(args.get("canvas_item_id") or args.get("item_id") or args.get("node_id") or "").strip()
    if not item_id:
        return _tool_error("canvas_item_id is required")
    if not args.get("canvas_id"):
        args["canvas_id"] = _ctx().get("canvas_id")
    url = _internal_api_url(f"canvas/nodes/{item_id}")
    if not url:
        return _tool_error("Canvas backend URL is not configured")
    try:
        resp = requests.patch(
            url,
            json=args,
            headers=_internal_relay_headers(),
            timeout=_backend_tool_timeout(60),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Canvas node update failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    if resp.status_code < 200 or resp.status_code >= 300:
        detail = decoded.get("detail") if isinstance(decoded, dict) else resp.text
        return _tool_error(str(detail or f"HTTP {resp.status_code}"))
    return json.dumps({"status": "success", "result": decoded}, ensure_ascii=False)


def _handle_canvas_connect_nodes(args: Dict[str, Any], **_: Any) -> str:
    scope_error = _require_canvas_graph_scope()
    if scope_error:
        return scope_error
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    args = dict(args or {})
    if not args.get("canvas_id"):
        args["canvas_id"] = _ctx().get("canvas_id")
    url = _internal_api_url("canvas/connections")
    if not url:
        return _tool_error("Canvas backend URL is not configured")
    try:
        resp = requests.post(
            url,
            json=args,
            headers=_internal_relay_headers(),
            timeout=_backend_tool_timeout(60),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Canvas connection create failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    if resp.status_code < 200 or resp.status_code >= 300:
        detail = decoded.get("detail") if isinstance(decoded, dict) else resp.text
        return _tool_error(str(detail or f"HTTP {resp.status_code}"))
    return json.dumps({"status": "success", "result": decoded}, ensure_ascii=False)


def _normalize_storybook_pages(value: Any) -> List[Dict[str, Any]]:
    """Return only valid page objects.

    Some models emit the pages array as a JSON string. Hermes' generic coercer
    may wrap a malformed string into a one-item list, which is still invalid for
    the Edu backend. For storybooks, malformed pages must fail planning rather
    than silently falling back to generic backend filler pages.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        decoded = _loads_storybook_pages_text(raw)
        if decoded is None:
            return []
        return _normalize_storybook_pages(decoded)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []

    pages: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            raw = item.strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(decoded, dict):
                item = decoded
            elif isinstance(decoded, list):
                pages.extend(_normalize_storybook_pages(decoded))
                continue
            else:
                continue
        if not isinstance(item, dict):
            continue
        page_number = item.get("page_number")
        page_index = item.get("page_index")
        try:
            if page_number is not None:
                item["page_number"] = int(page_number)
            if page_index is not None:
                item["page_index"] = int(page_index)
        except (TypeError, ValueError):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        if item.get("page_number") is None and item.get("page_index") is not None:
            item["page_number"] = int(item["page_index"]) + 1
        if item.get("page_index") is None and item.get("page_number") is not None:
            item["page_index"] = max(0, int(item["page_number"]) - 1)
        if item.get("page_number") is None:
            item["page_number"] = len(pages) + 1
            item["page_index"] = len(pages)
        if not str(item.get("page_type") or "").strip():
            item["page_type"] = "image"
        pages.append(item)
    return pages


def _loads_storybook_pages_text(raw: str) -> Any:
    text = _strip_json_fence(raw.strip())
    if not text:
        return None
    for candidate in _storybook_json_candidates(text):
        value = _loads_storybook_jsonish(candidate)
        if value is not None:
            return value
    objects = _extract_storybook_json_objects(text)
    if objects:
        return objects
    return None


def _loads_storybook_jsonish(raw: str, depth: int = 0) -> Any:
    if depth > 3:
        return None
    text = _strip_json_fence(str(raw or "").strip())
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (TypeError, ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            nested = _loads_storybook_jsonish(value, depth + 1)
            if nested is not None:
                return nested
            continue
        return value
    unquoted = _unquote_storybook_json_string(text)
    if unquoted and unquoted != text:
        return _loads_storybook_jsonish(unquoted, depth + 1)
    return None


def _unquote_storybook_json_string(text: str) -> str:
    text = text.strip()
    if len(text) < 2 or text[0] not in {'"', "'"} or text[-1] != text[0]:
        return text
    inner = text[1:-1]
    for encoding in ("utf-8",):
        try:
            return bytes(inner, encoding).decode("unicode_escape")
        except UnicodeDecodeError:
            continue
    return inner


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _storybook_json_candidates(text: str) -> List[str]:
    candidates = [text]
    unquoted = _unquote_storybook_json_string(text)
    if unquoted != text:
        candidates.append(unquoted)
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])
    if "[" in unquoted and "]" in unquoted:
        candidates.append(unquoted[unquoted.find("[") : unquoted.rfind("]") + 1])
    repaired = []
    for candidate in candidates:
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        fixed = re.sub(r"}\s*{", "},{", fixed)
        fixed = re.sub(r"}\s*\n\s*{", "},{", fixed)
        fixed = re.sub(r"]\s*{", "],{", fixed)
        fixed = re.sub(r'"\s*\n\s*"', '","', fixed)
        fixed = re.sub(
            r'("|\d|true|false|null|}|\])\s+(?=("[A-Za-z_][A-Za-z0-9_]*"\s*:))',
            r"\1,",
            fixed,
        )
        if fixed not in candidates and fixed not in repaired:
            repaired.append(fixed)
    return candidates + repaired


def _extract_storybook_json_objects(text: str) -> List[Dict[str, Any]]:
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        text = text[array_start + 1 : array_end]
    objects: List[Dict[str, Any]] = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                raw_obj = text[start : index + 1]
                parsed = _loads_storybook_object_text(raw_obj)
                if parsed is not None:
                    objects.append(parsed)
                start = -1
    return objects


def _loads_storybook_object_text(raw: str) -> Optional[Dict[str, Any]]:
    for candidate in _storybook_json_candidates(raw):
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            try:
                value = ast.literal_eval(candidate)
            except (TypeError, ValueError, SyntaxError):
                continue
        if isinstance(value, dict):
            return value
    return None


def _handle_alphart_create_storybook(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if not args.get("topic") and args.get("prompt"):
        args["topic"] = args.get("prompt")
    if not args.get("title") and args.get("topic"):
        args["title"] = args.get("topic")
    image_tool = _pick_tool("image", args)
    image_provider = str(image_tool.get("provider") or "").strip()
    image_model = str(image_tool.get("model") or image_tool.get("name") or image_tool.get("key") or "").strip()
    if image_provider.lower() != "openai" or "gpt-image" not in image_model.lower():
        image_provider = ""
        image_model = ""
    if image_provider:
        args.setdefault("image_provider", image_provider)
    if image_model:
        args.setdefault("image_model", image_model)
    args.setdefault("aspect_ratio", "1:1")
    args.setdefault("generate_images", True)
    if not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")
    create_url = _internal_api_url("storybooks")
    if not create_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    language = args.get("language") or _infer_storybook_language(args.get("topic") or args.get("prompt"))
    read_aloud_language = args.get("read_aloud_language") or _storybook_read_aloud_language(
        "\n".join(str(args.get(key) or "") for key in ("prompt", "topic", "description", "title")),
        language,
        _ctx().get("audio_language_type"),
    )
    payload = {
        "title": args.get("title"),
        "description": args.get("description"),
        "prompt": args.get("prompt"),
        "topic": args.get("topic") or args.get("prompt"),
        "language": language,
        "read_aloud_language": read_aloud_language,
        "age_range": args.get("age_range"),
        "reading_level": args.get("reading_level"),
        "style": args.get("style"),
        "page_count": args.get("page_count") or 10,
        "template_id": args.get("template_id"),
        "canvas_id": _ctx().get("canvas_id") or args.get("canvas_id"),
        "org_no": _ctx().get("org_no") or _ctx().get("storage_prefix") or args.get("org_no"),
    }
    raw_pages = args.get("pages")
    planned_pages = _normalize_storybook_pages(raw_pages)
    if raw_pages and not planned_pages:
        print(
            "[alphart-agent] storybook page plan was not valid JSON after repair; using backend physical-page planner",
            flush=True,
        )
    timeout = _backend_tool_timeout()
    storybook_id = ""
    created: Dict[str, Any] = {}
    try:
        created_resp = requests.post(create_url, json=payload, headers=_internal_relay_headers(), timeout=timeout)
        if created_resp.status_code < 200 or created_resp.status_code >= 300:
            return _tool_error(f"Storybook creation failed: {created_resp.text[:300]}")
        created = created_resp.json()
        storybook_id = str(created.get("id") or created.get("ID") or "").strip()
        if not storybook_id:
            return _tool_error("Storybook creation failed: missing id")
        plan_payload = {
            "topic": payload["topic"],
            "language": payload["language"],
            "read_aloud_language": read_aloud_language,
            "age_range": payload["age_range"],
            "reading_level": payload["reading_level"],
            "style": payload["style"],
            "page_count": payload["page_count"],
        }
        if planned_pages:
            plan_payload["pages"] = planned_pages
        plan_resp = requests.post(
            _internal_api_url(f"storybooks/{storybook_id}/plan"),
            json=plan_payload,
            headers=_internal_relay_headers(),
            timeout=timeout,
        )
        if plan_resp.status_code < 200 or plan_resp.status_code >= 300:
            return _tool_error(f"Storybook planning failed: {plan_resp.text[:300]}")
        generate_payload = {
            "generate_images": args.get("generate_images", True),
            "aspect_ratio": args.get("aspect_ratio") or "1:1",
            "image_provider": args.get("image_provider"),
            "image_model": args.get("image_model"),
            "input_images": args.get("input_images") or [],
        }
        gen_resp, generated = _generate_storybook_images_with_retries(storybook_id, generate_payload, timeout)
        if gen_resp.status_code < 200 or gen_resp.status_code >= 300:
            if not isinstance(generated, dict):
                generated = {}
            if not generated.get("pages"):
                generated = _fetch_storybook_after_generation_error(
                    storybook_id,
                    f"Storybook image generation failed: {gen_resp.text[:300]}",
                )
    except requests.RequestException as exc:
        if not storybook_id:
            return _tool_error(f"Storybook request failed: {exc}")
        return json.dumps(
            {
                "status": "success",
                "result": _partial_storybook_result(
                    storybook_id=storybook_id,
                    created=created if isinstance(created, dict) else {},
                    payload=payload,
                    generated={},
                    warning=f"Storybook request was interrupted after planning: {exc}",
                ),
            },
            ensure_ascii=False,
        )
    pages = generated.get("pages") if isinstance(generated, dict) else []
    storybook = generated.get("storybook") if isinstance(generated, dict) else {}
    image_report = generated.get("image_generation") if isinstance(generated, dict) else None
    generation_status = _storybook_result_status(image_report, pages)
    generation_warning = ""
    if isinstance(image_report, dict):
        required = int(image_report.get("required") or 0)
        missing = int(image_report.get("missing") or 0)
        errors = image_report.get("errors") or []
        if required > 0 and missing > 0:
            first_error = str(errors[0]) if errors else ""
            if len(first_error) > 180:
                first_error = first_error[:180].rstrip() + "..."
            generation_warning = f"{required - missing}/{required} illustrations generated, {missing} still pending."
            if first_error:
                generation_warning += f" {first_error}"
    result = {
        "type": "storybook",
        "status": generation_status,
        "warning": generation_warning,
        "presentation_mode": "flipbook",
        "read_aloud": True,
        "storybook_id": storybook_id,
        "title": storybook.get("title") or created.get("title") or payload.get("title"),
        "topic": storybook.get("topic") or created.get("topic") or payload.get("topic"),
        "page_count": len(pages or []),
        "canvas_id": payload.get("canvas_id"),
        "org_no": payload.get("org_no"),
        "pages": pages or [],
        "image_generation": image_report,
        "canvas_element": {
            "type": "embeddable",
            "link": "",
            "customData": {
                "kind": "storybook",
                "storybook_id": storybook_id,
                "title": storybook.get("title") or created.get("title") or payload.get("title"),
                "page_count": len(pages or []),
                "read_aloud": True,
                "mode": "flipbook",
                "pages": pages or [],
            },
        },
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _partial_storybook_result(
    *,
    storybook_id: str,
    created: Dict[str, Any],
    payload: Dict[str, Any],
    generated: Dict[str, Any],
    warning: str,
) -> Dict[str, Any]:
    pages = generated.get("pages") if isinstance(generated, dict) else []
    storybook = generated.get("storybook") if isinstance(generated, dict) else {}
    image_report = generated.get("image_generation") if isinstance(generated, dict) else None
    return {
        "type": "storybook",
        "status": _storybook_result_status(image_report, pages),
        "warning": warning,
        "presentation_mode": "flipbook",
        "read_aloud": True,
        "storybook_id": storybook_id,
        "title": storybook.get("title") or created.get("title") or payload.get("title"),
        "topic": storybook.get("topic") or created.get("topic") or payload.get("topic"),
        "page_count": len(pages or []),
        "canvas_id": payload.get("canvas_id"),
        "org_no": payload.get("org_no"),
        "pages": pages or [],
        "image_generation": image_report,
        "canvas_element": {
            "type": "embeddable",
            "link": "",
            "customData": {
                "kind": "storybook",
                "storybook_id": storybook_id,
                "title": storybook.get("title") or created.get("title") or payload.get("title"),
                "page_count": len(pages or []),
                "read_aloud": True,
                "mode": "flipbook",
                "pages": pages or [],
            },
        },
    }


def _generate_storybook_images_with_retries(storybook_id: str, payload: Dict[str, Any], timeout: int) -> Tuple[requests.Response, Dict[str, Any]]:
    attempts = max(1, int(os.getenv("ALPHART_STORYBOOK_IMAGE_RETRY_ATTEMPTS", "6") or "6"))
    last_resp: Optional[requests.Response] = None
    last_body: Dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                _internal_api_url(f"storybooks/{storybook_id}/generate"),
                json=payload,
                headers=_internal_relay_headers(),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_body = _fetch_storybook_after_generation_error(storybook_id, str(exc))
            last_resp = _synthetic_response(200 if not _storybook_image_report_has_missing(last_body.get("image_generation")) else 502, json.dumps(last_body, ensure_ascii=False))
            if not _storybook_image_report_has_missing(last_body.get("image_generation")) or attempt >= attempts:
                return last_resp, last_body
            time.sleep(min(2 * attempt, 5))
            continue
        last_resp = resp
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text}
        last_body = body if isinstance(body, dict) else {"raw": body}
        report = last_body.get("image_generation") if isinstance(last_body, dict) else None
        print(
            "[alphart-agent] storybook image generation report "
            f"storybook_id={storybook_id} attempt={attempt}/{attempts} status={resp.status_code} "
            f"report={json.dumps(report, ensure_ascii=False)}",
            flush=True,
        )
        if resp.status_code in {401, 403} or _storybook_image_report_has_permanent_failure(report):
            return resp, last_body
        if not _storybook_image_report_has_missing(report):
            return resp, last_body
        if attempt < attempts:
            time.sleep(min(2 * attempt, 5))
    return last_resp, last_body


def _synthetic_response(status_code: int, text: str) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = (text or "").encode("utf-8")
    return resp


def _fetch_storybook_after_generation_error(storybook_id: str, error: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "storybook": {"id": storybook_id},
        "pages": [],
        "image_generation": {"required": 0, "generated": 0, "skipped": 0, "missing": 0, "errors": [error]},
    }
    try:
        resp = requests.get(_internal_api_url(f"storybooks/{storybook_id}"), headers=_internal_relay_headers(), timeout=30)
        if resp.status_code < 200 or resp.status_code >= 300:
            body["image_generation"] = {"required": 0, "generated": 0, "skipped": 0, "missing": 0, "errors": [f"{error}; fetch status={resp.status_code}"]}
            return body
        fetched = resp.json()
    except (ValueError, requests.RequestException) as exc:
        body["image_generation"] = {"required": 0, "generated": 0, "skipped": 0, "missing": 0, "errors": [f"{error}; fetch failed: {exc}"]}
        return body
    if isinstance(fetched, dict):
        pages = fetched.get("pages") if isinstance(fetched.get("pages"), list) else []
        report = _storybook_image_report_from_pages(pages)
        if report.get("missing"):
            report["errors"] = [error]
        body = {"storybook": fetched.get("storybook") or fetched, "pages": pages, "image_generation": report}
    return body


def _storybook_image_report_from_pages(pages: Any) -> Dict[str, Any]:
    report = {"required": 0, "generated": 0, "skipped": 0, "missing": 0, "planned_pages": 0, "text_pages": 0}
    if not isinstance(pages, list):
        return report
    report["planned_pages"] = len(pages)
    for page in pages:
        if not isinstance(page, dict):
            continue
        if not _storybook_page_requires_image(page):
            if _storybook_page_has_content(page):
                report["text_pages"] += 1
            continue
        report["required"] += 1
        if str(page.get("image_s3_object_name") or page.get("image_url") or "").strip():
            report["generated"] += 1
        else:
            report["missing"] += 1
    return report


def _storybook_page_requires_image(page: Dict[str, Any]) -> bool:
    if not str(page.get("image_prompt") or "").strip():
        return False
    page_type = str(page.get("page_type") or "").strip().lower()
    return page_type not in {"narration", "text"}


def _storybook_page_has_content(page: Dict[str, Any]) -> bool:
    return bool(str(page.get("title") or "").strip() or str(page.get("narration") or "").strip())


def _storybook_result_status(report: Any, pages: Any) -> str:
    page_count = len(pages) if isinstance(pages, list) else 0
    text_pages = 0
    if isinstance(report, dict):
        try:
            required = int(report.get("required") or 0)
            generated = int(report.get("generated") or 0)
            missing = int(report.get("missing") or 0)
            text_pages = int(report.get("text_pages") or 0)
        except (TypeError, ValueError):
            required = generated = missing = 0
        if required > 0 and missing > 0:
            return "partial_finished" if generated > 0 or text_pages > 0 or page_count > 0 else "failed"
        if required > 0:
            return "completed"
    if page_count > 0:
        return "partial_finished"
    return "completed"


def _storybook_image_report_has_missing(report: Any) -> bool:
    if not isinstance(report, dict):
        return False
    try:
        required = int(report.get("required") or 0)
        missing = int(report.get("missing") or 0)
    except (TypeError, ValueError):
        return False
    return required > 0 and missing > 0


def _storybook_image_report_has_permanent_failure(report: Any) -> bool:
    if not isinstance(report, dict):
        return False
    for error in report.get("errors") or []:
        message = str(error or "").strip().lower()
        if any(
            marker in message
            for marker in (
                "status=401",
                "status=403",
                "code=unauthorized",
                "code=forbidden",
                "internal user uuid does not match",
            )
        ):
            return True
    return False


def _storybook_image_report_has_partial_success(report: Any) -> bool:
    if not isinstance(report, dict):
        return False
    try:
        generated = int(report.get("generated") or 0)
        missing = int(report.get("missing") or 0)
    except (TypeError, ValueError):
        return False
    return generated > 0 and missing > 0


def _handle_alphart_update_storybook_page(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")
    storybook_id = str(args.get("storybook_id") or args.get("id") or "").strip()
    if not storybook_id:
        return _tool_error("storybook_id is required")
    timeout = _backend_tool_timeout()
    page_id = str(args.get("page_id") or "").strip()
    if not page_id:
        try:
            resp = requests.get(_internal_api_url(f"storybooks/{storybook_id}"), headers=_internal_relay_headers(), timeout=timeout)
            if resp.status_code < 200 or resp.status_code >= 300:
                return _tool_error(f"Storybook page lookup failed: {resp.text[:300]}")
            pages = resp.json().get("pages") or []
            target_index = int(args.get("page_index") or ((int(args.get("page_number") or 1)) - 1))
            for page in pages:
                if isinstance(page, dict) and int(page.get("page_index") or -1) == target_index:
                    page_id = str(page.get("id") or "")
                    break
        except Exception as exc:
            return _tool_error(f"Storybook page lookup failed: {exc}")
    if not page_id:
        return _tool_error("storybook page not found")
    patch_payload = {
        "title": args.get("title"),
        "narration": args.get("narration"),
        "image_prompt": args.get("image_prompt") or args.get("instructions") or args.get("prompt"),
        "layout": args.get("layout"),
        "page_type": args.get("page_type"),
    }
    patch_payload = {k: v for k, v in patch_payload.items() if v is not None}
    try:
        resp = requests.patch(_internal_api_url(f"storybooks/{storybook_id}/pages/{page_id}"), json=patch_payload, headers=_internal_relay_headers(), timeout=timeout)
        if resp.status_code < 200 or resp.status_code >= 300:
            return _tool_error(f"Storybook page update failed: {resp.text[:300]}")
        regen_payload = {
            "generate_images": True,
            "aspect_ratio": args.get("aspect_ratio") or "1:1",
            "input_images": args.get("input_images") or [],
        }
        if args.get("image_provider"):
            regen_payload["image_provider"] = args.get("image_provider")
        if args.get("image_model"):
            regen_payload["image_model"] = args.get("image_model")
        regen = requests.post(_internal_api_url(f"storybooks/{storybook_id}/pages/{page_id}/regenerate"), json=regen_payload, headers=_internal_relay_headers(), timeout=timeout)
        if regen.status_code < 200 or regen.status_code >= 300:
            return _tool_error(f"Storybook page image regeneration failed: {regen.text[:300]}")
        payload = regen.json()
    except requests.RequestException as exc:
        return _tool_error(f"Storybook page update failed: {exc}")
    return json.dumps({"status": "success", "result": {"type": "storybook_page_update", "storybook_id": storybook_id, **payload}}, ensure_ascii=False)


def _handle_alphart_generate_image(args: Dict[str, Any], **kwargs: Any) -> str:
    args = dict(args or {})
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    if _canvas_explicit_video_request():
        return _tool_error(
            "Canvas video request already has a video workflow; use the supplied image references as keyframes instead of generating an image"
        )
    _set_canvas_model_default(args, "image")
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        if _ctx().get("image_aspect_ratio") and not args.get("aspect_ratio"):
            args["aspect_ratio"] = _ctx().get("image_aspect_ratio")
        elif _ctx().get("aspect_ratio") and not args.get("aspect_ratio"):
            args["aspect_ratio"] = _ctx().get("aspect_ratio")
        if _ctx().get("image_quality") and not args.get("quality"):
            args["quality"] = _ctx().get("image_quality")
        if _ctx().get("image_resolution") and not args.get("resolution"):
            args["resolution"] = _ctx().get("image_resolution")
    if args.get("image_quantity") and not args.get("quantity"):
        args["quantity"] = args.get("image_quantity")
    if not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")
    tool = _pick_tool("image", args)
    args.setdefault("provider", tool.get("provider"))
    args.setdefault("model", tool.get("model") or tool.get("name") or tool.get("key"))
    # Canvas owns a generation-only relay. Reference nodes remain structured
    # Canvas context for prompt composition; do not send them to Edu's edit API.
    is_canvas = str(_ctx().get("app_scope") or "").strip().lower() == "canvas"
    explicit_canvas_item_id = str(_ctx().get("canvas_item_id") or "").strip()
    selected_canvas_item_id = str(_ctx().get("selected_canvas_item_id") or "").strip()
    selected_canvas_item_type = str(_ctx().get("selected_canvas_item_type") or "").strip().lower()
    candidate_item_id = str(
        args.get("canvas_item_id") or args.get("item_id") or args.get("node_id") or ""
    ).strip()
    if (
        is_canvas
        and not explicit_canvas_item_id
        and candidate_item_id
        and candidate_item_id == selected_canvas_item_id
        and selected_canvas_item_type in {"text", "note"}
    ):
        # A selected text node is graph context, never the image execution target.
        for key in ("canvas_item_id", "item_id", "node_id"):
            args.pop(key, None)
    if is_canvas and not str(args.get("canvas_item_id") or "").strip() and not str(_ctx().get("canvas_item_id") or "").strip():
        image_item_id, graph_error = _ensure_canvas_image_generation_graph(str(args.get("prompt") or "").strip())
        if graph_error:
            return _tool_error(graph_error)
        if image_item_id:
            args["canvas_item_id"] = image_item_id
    use_seedream_sdk = _jwell_relay_enabled() and _is_seedream_image_model(
        args.get("provider"), args.get("model")
    )
    relay_url = _relay_url(
        "images/generations"
        if is_canvas or not args.get("input_images") or use_seedream_sdk
        else "images/edits"
    )
    if not relay_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    payload = {
        "model": args.get("model"),
        "provider": args.get("provider"),
        "prompt": args.get("prompt"),
        "aspect_ratio": args.get("aspect_ratio"),
        "quality": args.get("quality"),
        "resolution": args.get("resolution"),
        "session_id": _ctx().get("session_id"),
        "canvas_id": _ctx().get("canvas_id"),
        "canvas_item_id": (
            args.get("canvas_item_id")
            or args.get("item_id")
            or args.get("node_id")
            or _ctx().get("canvas_item_id")
            or _latest_canvas_created_node_id("image")
        ),
        "tool_call_id": kwargs.get("tool_call_id") or args.get("tool_call_id"),
    }
    if args.get("quantity"):
        payload["n"] = args.get("quantity")
    if args.get("input_images"):
        try:
            payload["images"] = _resolve_jwell_media_urls("image", args.get("input_images"))
        except (requests.RequestException, RuntimeError) as exc:
            return _tool_error(f"Unable to prepare image references: {exc}")
    if is_canvas:
        payload["reference_item_ids"] = _ctx().get("reference_item_ids") or []
    print(
        f"[alphart-agent] calling internal relay image session_id={_ctx().get('session_id')} url={relay_url}",
        flush=True,
    )
    response_status = 0
    response_preview = ""
    if use_seedream_sdk:
        try:
            from tools.seedream_sdk import create_seedream_image

            decoded = create_seedream_image(
                base_url=f"{_jwell_relay_base_url()}/internal/v1",
                headers=_relay_headers(payload.get("tool_call_id")),
                model=str(args.get("model") or ""),
                prompt=str(args.get("prompt") or ""),
                provider=str(args.get("provider") or ""),
                image_urls=payload.get("images") or [],
                aspect_ratio=str(args.get("aspect_ratio") or ""),
                resolution=str(args.get("resolution") or ""),
                quantity=args.get("quantity"),
                idempotency_key=str(payload.get("tool_call_id") or "").strip(),
                timeout=_backend_tool_timeout(),
            )
            response_status = 200
            response_preview = json.dumps(decoded, ensure_ascii=False)[:500]
        except Exception as exc:  # SDK errors must remain tool failures, not agent crashes.
            return _tool_error(f"Alphart relay request failed: {exc}")
    else:
        try:
            resp = requests.post(
                relay_url,
                json=payload,
                headers=_relay_headers(payload.get("tool_call_id")),
                timeout=_backend_tool_timeout(),
            )
        except requests.RequestException as exc:
            return _tool_error(f"Alphart relay request failed: {exc}")
        response_status = resp.status_code
        try:
            decoded = resp.json()
        except ValueError:
            decoded = {"raw": resp.text}
        response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] internal relay image response status={response_status} body={response_preview}",
        flush=True,
    )
    if response_status < 200 or response_status >= 300:
        if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
            detail = ""
            if isinstance(decoded, dict):
                error = decoded.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("detail") or "")
                detail = detail or str(decoded.get("detail") or decoded.get("message") or "")
            detail = " ".join((detail or f"relay returned HTTP {response_status}").split())[:500]
            return _tool_error(f"Canvas image relay failed (HTTP {response_status}): {detail}")
        return _system_busy_tool_error()
    if is_canvas and not str(_ctx().get("canvas_item_id") or "").strip():
        _ctx()["_canvas_generation_submitted"] = True
    data = decoded.get("data") if isinstance(decoded, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        asset = data[0]
    elif isinstance(data, dict):
        asset = data
    elif isinstance(decoded, dict) and isinstance(decoded.get("result"), dict):
        asset = decoded["result"]
    elif isinstance(decoded, dict):
        asset = decoded
    else:
        asset = {}
    if _jwell_relay_enabled():
        try:
            asset = _import_jwell_media(asset, "image")
        except (requests.RequestException, RuntimeError) as exc:
            return _tool_error(f"Unable to store generated image: {exc}")
    result = {
        "type": "image",
        "provider": asset.get("provider") or args.get("provider"),
        "model": asset.get("model") or args.get("model"),
        "url": asset.get("url") or asset.get("result_image_url"),
        "mime_type": asset.get("mime_type"),
        "width": asset.get("width"),
        "height": asset.get("height"),
        "filename": asset.get("filename"),
        "s3_object_name": asset.get("s3_object_name") or asset.get("object_key") or asset.get("result_image_object_key"),
        "usage": asset.get("usage"),
        "_credit_settled": bool(asset.get("_credit_settled")),
        "_credit_reference_id": asset.get("_credit_reference_id"),
    }
    if is_canvas and not str(result.get("s3_object_name") or "").strip():
        return _tool_error("Canvas image relay returned no stored image asset")
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _handle_alphart_generate_video(args: Dict[str, Any], **kwargs: Any) -> str:
    args = dict(args or {})
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    _set_canvas_model_default(args, "video")
    is_canvas = str(_ctx().get("app_scope") or "").strip().lower() == "canvas"
    if not str(args.get("prompt") or "").strip() and is_canvas:
        args["prompt"] = str(_ctx().get("canvas_prompt_context") or _ctx().get("user_message") or "").strip()
    if is_canvas and _ctx().get("aspect_ratio") and not args.get("aspect_ratio"):
        args["aspect_ratio"] = _ctx().get("aspect_ratio")
    if is_canvas and _ctx().get("resolution") and not args.get("resolution"):
        args["resolution"] = _ctx().get("resolution")
    # Tool arguments are the LLM's explicit decision. Canvas options are the
    # fallback when the model omitted a duration, followed by the request text
    # parser and the five-second default.
    requested_duration = _positive_video_duration(args.get("duration"))
    if requested_duration is None:
        requested_duration = _positive_video_duration(args.get("duration_seconds"))
    if requested_duration is None and is_canvas:
        requested_duration = _positive_video_duration(_ctx().get("duration_seconds"))
    if requested_duration is None:
        request_text = _ctx().get("user_message") or _ctx().get("canvas_prompt_context")
        requested_duration = _video_duration_seconds_from_text(request_text) or 5
    args["duration"] = requested_duration
    if args.get("image_url") and not args.get("input_images"):
        args["input_images"] = [args.get("image_url")]
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        # Connected Canvas nodes are the user-selected keyframes. Keep them
        # ahead of any model-suggested images so an audio reference can never
        # reach the video relay without its visual counterpart.
        canvas_images = [
            entry for entry in (_ctx().get("input_images") or [])
            if _canvas_image_reference_key(entry)
        ]
        requested_images = [
            entry for entry in (args.get("input_images") or [])
            if _canvas_image_reference_key(entry)
        ]
        merged_images = list(canvas_images)
        known_images = {_canvas_image_reference_key(entry) for entry in canvas_images}
        for entry in requested_images:
            identity = _canvas_image_reference_key(entry)
            if identity and identity not in known_images:
                merged_images.append(entry)
                known_images.add(identity)
        args["input_images"] = merged_images
    elif not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")

    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        canvas_audio = [entry for entry in (_ctx().get("input_audio") or []) if isinstance(entry, dict)]
        requested_audio = [entry for entry in (args.get("input_audio") or []) if isinstance(entry, dict)]
        # Directly connected Canvas audio is authoritative. Preserve any additional
        # model-selected references without allowing it to discard the soundtrack.
        merged_audio = list(canvas_audio)
        known_audio = {
            str(entry.get("s3_object_name") or entry.get("object_key") or entry.get("url") or "")
            for entry in canvas_audio
        }
        for entry in requested_audio:
            identity = str(entry.get("s3_object_name") or entry.get("object_key") or entry.get("url") or "")
            if identity and identity in known_audio:
                continue
            merged_audio.append(entry)
        args["input_audio"] = merged_audio
    elif not args.get("input_audio") and _ctx().get("input_audio"):
        args["input_audio"] = _ctx().get("input_audio")
    has_canvas_soundtrack = (
        str(_ctx().get("app_scope") or "").strip().lower() == "canvas"
        and any(
            str((entry or {}).get("role") or "").strip().lower() in {"soundtrack", "background_music"}
            for entry in (args.get("input_audio") or [])
            if isinstance(entry, dict)
        )
    )
    if has_canvas_soundtrack:
        args["generate_audio"] = False
    elif str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        # The user's explicit audio instruction wins over the UI/default. When
        # the prompt is silent, keep the selected Canvas option as the fallback.
        preference = _explicit_audio_preference(_ctx().get("user_message"))
        if preference is not None:
            args["generate_audio"] = preference
        elif "generate_audio" not in args:
            args["generate_audio"] = bool(_ctx().get("generate_audio"))
    elif "generate_audio" not in args:
        args["generate_audio"] = bool(_ctx().get("generate_audio"))
    tool = _pick_tool("video", args)
    _set_tool_defaults(args, tool)
    args.setdefault("wait", False)
    if is_canvas:
        # A model-supplied node id is not authoritative. In a homepage/new
        # graph turn it may be stale or hallucinated, and trusting it skips
        # the graph creation fallback before the relay validates the node.
        # Only the selected execution target or a node created in this turn
        # can be used for Canvas generation. Edu keeps its existing argument
        # precedence and relay contract.
        context_item_id = _canvas_context_video_item_id()
        selected_item_id = str(_ctx().get("selected_canvas_item_id") or "").strip()
        selected_item_type = str(_ctx().get("selected_canvas_item_type") or "").strip().lower()
        created_item_id = "" if _ctx().get("_canvas_generation_submitted") else _latest_canvas_created_node_id("video")
        canvas_item_id = context_item_id or (
            selected_item_id if selected_item_type == "video" else ""
        ) or created_item_id
    else:
        canvas_item_id = str(
            args.get("canvas_item_id")
            or args.get("item_id")
            or args.get("node_id")
            or _ctx().get("canvas_item_id")
            or ""
        ).strip()
    auto_created_video_node = False
    if is_canvas and not canvas_item_id:
        canvas_item_id, graph_error = _ensure_canvas_video_generation_graph(str(args.get("prompt") or "").strip())
        if graph_error:
            return _tool_error(graph_error)
        args["canvas_item_id"] = canvas_item_id
        auto_created_video_node = bool(canvas_item_id)
    relay_url = _relay_url("videos", provider=args.get("provider"), model=args.get("model"))
    if not relay_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    payload = {
        "model": args.get("model"),
        "provider": args.get("provider"),
        "prompt": args.get("prompt"),
        "aspect_ratio": args.get("aspect_ratio"),
        "resolution": args.get("resolution"),
        "duration": args.get("duration"),
        "session_id": _ctx().get("session_id"),
        "canvas_id": _ctx().get("canvas_id"),
        "canvas_item_id": canvas_item_id,
        "generate_audio": bool(args.get("generate_audio")),
        "tool_call_id": kwargs.get("tool_call_id") or args.get("tool_call_id"),
    }
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        payload.update({
            # The tool argument is authoritative when the model supplied one;
            # the request-level value covers the selected-node composer path.
            "caption_script": str(args.get("caption_script") or _ctx().get("video_caption_script") or "").strip(),
            "audio_model": _ctx().get("audio_model"),
            "language_type": _ctx().get("audio_language_type"),
            "create_if_missing": auto_created_video_node,
        })
        payload.update({
            "user_id": _ctx().get("user_id"),
            "user_uuid": _ctx().get("user_uuid"),
            "org_no": _ctx().get("org_no"),
            "reference_item_ids": _ctx().get("reference_item_ids") or [],
        })
    if args.get("input_images"):
        images = args.get("input_images")
        # Edu's video relay accepts image-to-video references under `image`;
        # the value itself may be a list for multi-reference providers.
        try:
            payload["image"] = _resolve_jwell_media_urls("image", images)
        except (requests.RequestException, RuntimeError) as exc:
            return _tool_error(f"Unable to prepare video image references: {exc}")
    if args.get("input_audio"):
        try:
            payload["audio"] = _resolve_jwell_media_urls("audio", args.get("input_audio"))
        except (requests.RequestException, RuntimeError) as exc:
            return _tool_error(f"Unable to prepare video audio references: {exc}")
    print(
        f"[alphart-agent] calling internal relay video session_id={_ctx().get('session_id')} "
        f"provider={_log_model_value(payload.get('provider'))} "
        f"model={_log_model_value(payload.get('model'))} url={relay_url}",
        flush=True,
    )
    try:
        if _jwell_relay_enabled() and _is_seedance_video_model(payload.get("provider"), payload.get("model")):
            from tools.seedance_sdk import create_seedance_task

            sdk_headers = _relay_headers()
            tool_call_id = str(payload.get("tool_call_id") or "").strip()
            if tool_call_id:
                sdk_headers["Idempotency-Key"] = tool_call_id
            image_references = args.get("input_images") or []
            audio_references = args.get("input_audio") or []
            decoded = create_seedance_task(
                base_url=_jwell_relay_base_url() + "/internal/v3",
                headers=sdk_headers,
                model=str(payload.get("model") or "").strip(),
                prompt=str(payload.get("prompt") or ""),
                image_urls=payload.get("image") or [],
                audio_urls=payload.get("audio") or [],
                image_roles=_seedance_media_roles(image_references),
                audio_roles=_seedance_media_roles(audio_references),
                ratio=str(payload.get("aspect_ratio") or "").strip(),
                resolution=str(payload.get("resolution") or "").strip(),
                duration=int(payload["duration"]) if payload.get("duration") is not None else None,
                generate_audio=bool(payload.get("generate_audio")),
                timeout=_backend_tool_timeout(),
            )
            response_status = 202
            response_preview = json.dumps(decoded, ensure_ascii=False)[:500]
            print(
                f"[alphart-agent] official Ark SDK relay response status={response_status} "
                f"body={response_preview}",
                flush=True,
            )
        else:
            resp = requests.post(
                relay_url,
                json=payload,
                headers=_relay_headers(payload.get("tool_call_id")),
                timeout=_backend_tool_timeout(),
            )
            try:
                decoded = resp.json()
            except ValueError:
                decoded = {"raw": resp.text}
            response_preview = (resp.text or "").replace("\n", " ")[:500]
            print(
                f"[alphart-agent] internal relay video response status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
                flush=True,
            )
            response_status = resp.status_code
    except Exception as exc:  # SDK provider errors are version-specific classes.
        print(
            f"[alphart-agent] Seedance video relay failed session_id={_ctx().get('session_id')} "
            f"error_type={type(exc).__name__} error={exc}",
            flush=True,
        )
        if is_canvas:
            return _tool_error(f"Canvas video relay failed: {exc}")
        return _system_busy_tool_error()
    if response_status < 200 or response_status >= 300:
        if is_canvas:
            detail = ""
            if isinstance(decoded, dict):
                detail = str(decoded.get("detail") or decoded.get("message") or decoded.get("error") or "").strip()
            return _tool_error(
                f"Canvas video relay failed (HTTP {response_status}): "
                f"{detail or response_preview or 'unknown relay error'}"
            )
        return _system_busy_tool_error()
    if is_canvas:
        # A successful relay submission completes this automatic graph's
        # lifecycle. The next automatic request must create a fresh pair
        # instead of updating the previous video node.
        _ctx()["_canvas_generation_submitted"] = True
    selected_provider = decoded.get("provider") if isinstance(decoded, dict) else args.get("provider")
    selected_model = decoded.get("model") if isinstance(decoded, dict) else args.get("model")
    task_id = ""
    if isinstance(decoded, dict):
        # Canvas may return an in-flight local task while the original request
        # is still submitting and no provider task id exists yet. Edu/Jwell
        # responses continue to use the provider id as before.
        task_id = decoded.get("id") or decoded.get("task_id") or ""
    if _jwell_relay_enabled():
        try:
            tracked = requests.post(
                _internal_api_url("agent/jwell-video-tasks"),
                json={
                    "task_id": task_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "resolution": payload.get("resolution"),
                    "canvas_id": payload.get("canvas_id"),
                    "session_id": payload.get("session_id"),
                    "tool_call_id": payload.get("tool_call_id"),
                },
                headers=_internal_relay_headers(),
                timeout=_backend_tool_timeout(),
            )
            tracked.raise_for_status()
        except requests.RequestException as exc:
            return _tool_error(f"Unable to track Jwell video task: {exc}")
    print(
        f"[alphart-agent] internal relay video selected "
        f"provider={_log_model_value(selected_provider)} model={_log_model_value(selected_model)}",
        flush=True,
    )
    result = {
        "phase": "task",
        "type": "generate_video_task",
        "status": decoded.get("status") if isinstance(decoded, dict) else "queued",
        "message": "Create generation task success",
        "task_id": task_id,
        "provider": selected_provider,
        "model": selected_model,
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _handle_alphart_generate_audio(args: Dict[str, Any], **kwargs: Any) -> str:
    args = dict(args or {})
    mutation_error = _require_canvas_mutation_allowed()
    if mutation_error:
        return mutation_error
    audio_chunk_request = bool(args.pop("_audio_chunk", False))
    skip_media_import = bool(args.pop("_skip_media_import", False))
    _set_canvas_model_default(args, "audio")
    canvas_audio_duration = _ctx().get("audio_duration_seconds") if _ctx().get("app_scope") == "canvas" else 0
    requested_duration = int(args.get("duration_seconds") or canvas_audio_duration or _ctx().get("duration_seconds") or 0)
    if _ctx().get("app_scope") == "canvas":
        requested_duration = max(5, min(15, requested_duration or 5))
    tool = _pick_tool("audio", args)
    _set_tool_defaults(args, tool)
    selected_provider = str(args.get("provider") or "").strip()
    selected_model = str(args.get("model") or "").strip()
    if not selected_provider or not selected_model:
        return _tool_error("No configured audio generation model is available.", "AUDIO_MODEL_NOT_CONFIGURED")
    args["language_type"] = _normalize_audio_language_type(args.get("language_type")) or _normalize_audio_language_type(_ctx().get("audio_language_type"))
    relay_url = _relay_url("audio/speech")
    if not relay_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    text = str(args.get("input") or args.get("text") or args.get("script") or args.get("prompt") or "").strip()
    approved_script = str(_ctx().get("approved_audio_script") or "").strip()
    if _ctx().get("app_scope") == "canvas" and approved_script:
        text = approved_script
    if not text:
        return _tool_error("audio input text is required")
    if _ctx().get("app_scope") != "canvas" and (len(text) < 80 or re.search(r"^\s*(/audio|generate|create|生成)", text, flags=re.I)):
        text = _audio_script_from_request(text, str(args.get("language_type") or ""))
    if _jwell_relay_enabled() and not audio_chunk_request:
        chunks = _split_audio_script(text, str(args.get("language_type") or ""))
        if len(chunks) > 1:
            response_format = str(args.get("response_format") or "wav").strip().lower()
            if response_format not in {"wav", "wave", "x-wav", "audio/wav", "audio/x-wav"}:
                return _tool_error(
                    "Long audio chunking requires response_format=wav",
                    "AUDIO_FORMAT_UNSUPPORTED",
                )
            print(
                f"[alphart-agent] audio chunking provider={_log_model_value(selected_provider)} "
                f"model={_log_model_value(selected_model)} chunks={len(chunks)}",
                flush=True,
            )
            return _generate_chunked_audio(args, text, chunks, kwargs.get("tool_call_id") or "")
    payload = {
        "provider": selected_provider,
        "model": args.get("model"),
        "input": text,
        "voice": args.get("voice"),
        "language_type": args.get("language_type"),
        "response_format": args.get("response_format") or "wav",
        "duration_seconds": requested_duration or None,
        "session_id": _ctx().get("session_id"),
        "canvas_id": _ctx().get("canvas_id"),
        "canvas_item_id": _ctx().get("canvas_item_id"),
        "user_id": _ctx().get("user_id"),
        "user_uuid": _ctx().get("user_uuid"),
        "org_no": _ctx().get("org_no"),
        "idempotency_key": kwargs.get("tool_call_id") or args.get("tool_call_id"),
    }
    if _ctx().get("app_scope") == "canvas":
        payload["reference_item_ids"] = _ctx().get("reference_item_ids") or []
    native_route = (
        "gemini.generateContent"
        if str(selected_provider).strip().lower() in {"google", "gemini"}
        else "audio.speech"
    )
    print(
        f"[alphart-agent] calling internal relay audio session_id={_ctx().get('session_id')} "
        f"provider={_log_model_value(selected_provider)} model={_log_model_value(selected_model)} "
        f"route={native_route} url={relay_url}",
        flush=True,
    )
    attempts = max(1, int(os.getenv("ALPHART_AUDIO_RELAY_RETRY_ATTEMPTS", "3") or "3"))
    timeout = _backend_tool_timeout()
    resp = None
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"[alphart-agent] internal relay audio attempt session_id={_ctx().get('session_id')} "
                f"attempt={attempt}/{attempts}",
                flush=True,
            )
            resp = requests.post(
                relay_url,
                json=payload,
                headers=_relay_headers(payload.get("idempotency_key")),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            print(
                f"[alphart-agent] internal relay audio request failed "
                f"attempt={attempt}/{attempts} error={exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(2 * attempt, 5))
                continue
            return _tool_error(f"Alphart relay request failed: {exc}")
        if resp.status_code not in {502, 503, 504} or attempt >= attempts:
            break
        response_preview = (resp.text or "").replace("\n", " ")[:500]
        print(
            f"[alphart-agent] internal relay audio retryable response "
            f"attempt={attempt}/{attempts} status={resp.status_code} body={response_preview}",
            flush=True,
        )
        time.sleep(min(2 * attempt, 5))
    if resp is None:
        return _tool_error(f"Alphart relay request failed: {last_exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] internal relay audio response status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
        flush=True,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        if isinstance(decoded, dict) and isinstance(decoded.get("result"), dict):
            result = dict(decoded["result"])
            result.setdefault("type", "generate_audio_result")
            result.setdefault("status", "partial" if result.get("usage") else "failed")
            result.setdefault("message", decoded.get("message") or "generate fail")
            result.setdefault("provider", selected_provider)
            result.setdefault("model", selected_model)
            result.setdefault("input", text)
            result.setdefault("script", text)
            return json.dumps({"status": "failed", "result": result}, ensure_ascii=False)
        return _system_busy_tool_error()
    data = decoded.get("data") if isinstance(decoded, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        asset = data[0]
    elif isinstance(data, dict):
        asset = data
    elif isinstance(decoded, dict) and isinstance(decoded.get("result"), dict):
        asset = decoded["result"]
    elif isinstance(decoded, dict):
        asset = decoded
    else:
        asset = {}
    if _jwell_relay_enabled() and not skip_media_import:
        try:
            asset = _import_jwell_media(asset, "audio")
        except (requests.RequestException, RuntimeError) as exc:
            return _tool_error(f"Unable to store generated audio: {exc}")
    audio_url = asset.get("url") or asset.get("audio_url")
    if not audio_url:
        return _system_busy_tool_error()
    selected_provider = asset.get("provider") or args.get("provider")
    selected_model = asset.get("model") or args.get("model")
    print(
        f"[alphart-agent] internal relay audio selected "
        f"provider={_log_model_value(selected_provider)} model={_log_model_value(selected_model)}",
        flush=True,
    )
    result = {
        "type": "generate_audio_result",
        "provider": selected_provider,
        "model": selected_model,
        "input": text,
        "script": text,
        "url": audio_url,
        "audio_url": audio_url,
        "mime_type": asset.get("mime_type") or "audio/wav",
        "duration_seconds": asset.get("duration_seconds"),
        "filename": asset.get("filename"),
        "s3_object_name": asset.get("s3_object_name"),
        "usage": asset.get("usage"),
        "_credit_settled": bool(asset.get("_credit_settled")),
        "_credit_reference_id": asset.get("_credit_reference_id"),
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _default_game_plan() -> Dict[str, Any]:
    return {
        "default_style": "simple pixel game, not a web form",
        "preferred_patterns": [
            "side-scrolling collect-and-dodge platformer",
            "top-down maze/exploration game",
            "drag-and-drop sorting challenge with moving sprites",
            "timed arcade matcher",
            "mini simulation sandbox with cause/effect meters",
            "boss/level challenge where answers power attacks or shields",
            "physics launcher/trajectory challenge",
            "story quest with rooms, NPC prompts, and collectible facts",
        ],
        "steps": [
            "Studio Design phase: define the learning goal, audience, player fantasy, target emotion, and winning objective.",
            "Extract a content_facts ledger: exact facts, formulas, definitions, vocabulary, units, correct answers, answer option groups, constraints, and misconceptions that must be preserved.",
            "Choose a playable pixel-game pattern that fits the concept; avoid a static form or plain card quiz unless explicitly requested.",
            "Map learning content into player actions, hazards, collectibles, levels, feedback, and win/fail conditions.",
            "Studio Planning phase: turn the design into acceptance criteria for layout bounds, controls, core loop, scoring/progress, validation/collision, and completion states.",
            "Studio Development phase: implement an exact bounded 1920x1080 logical playfield that scales to the iframe viewport, HUD, safe text panels, start/restart controls, and separated game state/update/render logic.",
            "Generate one complete self-contained HTML file with inline CSS/JS and no external assets.",
            "Studio QA phase: self-test that no widget/window overflows, controls mutate state, the loop runs, scoring/progress changes, and win/fail/completion is reachable before finalizing.",
        ],
    }


def _default_game_layout_requirements() -> Dict[str, Any]:
    return {
        "logical_width": 1920,
        "logical_height": 1080,
        "responsive": "The game must use an exact fixed 1920x1080 logical stage and scale that whole stage to fit smaller browser/iframe viewports without page scroll.",
        "visual_style": "Default to a simple pixel-art game: blocky sprites, tile/grid playfield, crisp edges, limited high-contrast palette, 8-bit inspired UI.",
        "anti_form_rule": "Do not make a plain web form, static worksheet, or button-only quiz unless the user explicitly asks for a quiz/form.",
        "playfield": "Use a real game area with player movement, collectibles/hazards/targets, score/progress, and visible feedback.",
        "stage_rule": "The result page must include one visible root game stage with width:1920px and height:1080px. Center and scale that exact stage to fit the viewport; do not make the document or stage larger than 1920x1080.",
        "safe_area": "Keep all text, buttons, sprites, score panels, dialogs, modals, tooltips, and windows inside x=40..1880 and y=40..1040 of the 1920x1080 stage.",
        "overflow_policy": "No clipped text, page scrolling, horizontal scroll, overlapping cards, or elements outside their parent border. Use box-sizing:border-box globally and overflow:hidden on html, body, and the root stage.",
        "positioning_policy": "Absolutely positioned panels/windows/widgets must use explicit left/top/width/height values that fit within the safe area including padding and borders. Do not use negative offsets, fixed-position overlays, oversized absolute dialogs, viewport-sized panels, or transforms that push UI out of frame.",
        "typography": "Use readable font sizes and wrap long labels instead of shrinking below legibility.",
        "controls": "Support keyboard and mouse/touch. On-screen controls must remain reachable on desktop and mobile sizes.",
    }


def _default_game_review_checklist() -> Dict[str, Any]:
    return {
        "content": [
            "Matches the user's requested topic and age/education level.",
            "Preserves factual precision: formulas, units, definitions, dates, names, symbols, vocabulary, and causal relationships must be correct.",
            "Every correct answer, wrong answer, collectible, hazard, label, dialog, and feedback message traces to the user request, game_plan knowledge_points/misconceptions, or stable common knowledge.",
            "Every question or answer-option group includes at least one correct choice. Single-answer groups have exactly one correct choice; multi-select groups clearly say multi-select and have one or more correct choices.",
            "No impossible question is shown. If no correct option can be derived, replace it with a non-scored explanation or a different valid question.",
            "No unsupported numbers, equations, dates, named entities, properties, or causal claims are invented just to make gameplay work.",
            "No oversimplification that changes meaning; if simplifying for children, keep the core concept technically accurate.",
            "Wrong answers, hazards, and feedback must teach the precise correction explicitly and must not make a misconception sound true.",
            "Includes clear instructions and a measurable goal or win condition.",
            "Does not add unrelated facts, unsafe content, or confusing filler.",
            "No vulgar/profane language, sexual content, graphic violence, gore, hate/harassment, self-harm, drug abuse, gambling, extremist content, humiliation, or age-inappropriate jokes.",
            "Combat, enemies, attacks, hazards, and penalties are represented with non-graphic classroom-safe metaphors such as shields, puzzles, energy, obstacles, misconception blockers, or abstract pixel effects.",
        ],
        "ui_layout": [
            "No element exceeds the viewport, card, panel, or border.",
            "Every HUD, modal/window, widget, card, dialog, button, tooltip, label, and sprite stays inside x=40..1880 and y=40..1040 of the 1920x1080 stage.",
            "No text is clipped or hidden behind another element.",
            "No page scrolling is possible; html/body/stage overflow is hidden and layout is contained.",
            "No negative offsets, fixed-position overlays, viewport-sized panels, or transforms can push UI outside the stage.",
            "Buttons, score, progress, dialogs, and game objects have enough spacing.",
            "The layout works at 1920x1080, 1366x768, 1024x768, and phone-sized iframe viewports without clipped controls, hidden text, overlap, or scroll.",
        ],
        "interaction": [
            "Start/restart flow works.",
            "Start/restart buttons have real event handlers, not placeholder UI.",
            "There is an actual play loop: move/choose/collect/avoid/solve, then receive immediate feedback.",
            "Keyboard, mouse, touch, or button input mutates state and updates visible UI.",
            "Player can win, fail, or complete a level; the state is visible.",
            "Win/fail/completion state is visible.",
            "Keyboard, mouse, and touch interactions are not blocked.",
        ],
        "playability": [
            "Not just a form or static card.",
            "At least one sprite/player/object moves or changes in response to input.",
            "Challenge has pacing: timer, level, score target, hazards, or progression.",
            "Score, progress, timer, level, selected answer, or player position changes through gameplay.",
            "Collision, answer checking, target validation, or equivalent challenge logic is implemented.",
            "Win/fail/completion can be reached by playing the game.",
            "No TODO placeholders, stub handlers, fake buttons, or comments replacing core game logic.",
        ],
    }


def _game_artifact_harness_feedback(html: str) -> str:
    """Return the first actionable failure from the 2D game upload harness.

    The agent cannot reliably visually inspect every generated document before
    uploading it. Keep this gate structural and deterministic: it validates the
    self-contained artifact contract and the minimum signals of a playable 2D
    game without trying to prescribe one particular game implementation.
    """
    value = str(html or "").strip()
    lower = value.lower()
    if not value:
        return "game artifact harness: HTML is empty"
    if not lower.startswith("<!doctype html"):
        return "game artifact harness: HTML must start with <!DOCTYPE html>"
    if "<body" not in lower:
        return "game artifact harness: HTML must include a <body> with visible content"
    if "</body>" not in lower:
        return "game artifact harness: HTML is truncated: missing </body>"
    if "</html>" not in lower:
        return "game artifact harness: HTML is truncated: missing </html>"
    if re.search(r"<(?:script|link)\b[^>]+(?:src|href)\s*=\s*['\"]https?://", value, re.I):
        return "game artifact harness: external scripts or stylesheets are not allowed; include all game code locally"
    if re.search(r"<(?:img|audio|video|source)\b[^>]+src\s*=\s*['\"]https?://", value, re.I):
        return "game artifact harness: external media assets are not allowed; use inline or local artifact assets"
    if "<script" not in lower and "onclick=" not in lower and "addeventlistener" not in lower:
        return "game artifact harness: game must include inline JavaScript interaction code"
    body_start = lower.find("<body")
    body_end = lower.rfind("</body>")
    body_open_end = lower.find(">", body_start)
    if body_start < 0 or body_end <= body_start or body_open_end < 0:
        return "game artifact harness: HTML must include a valid <body> element"
    body = value[body_open_end + 1 : body_end].strip()
    visible_body = _strip_invisible_game_html(body)
    if not visible_body.strip():
        return "game artifact harness: body must include visible game DOM content directly in <body>"
    if not re.search(r"<(main|section|article|div|canvas|svg|button|input|label|h[1-6]|p|span)\b", visible_body, re.I | re.S):
        return "game artifact harness: body must include visible game DOM elements directly in <body>"
    if not re.search(r"<(canvas|svg)\b", visible_body, re.I):
        return "game artifact harness: game must include a visible Canvas or SVG playfield"
    if not re.search(r"<(button|input)\b", visible_body, re.I):
        return "game artifact harness: game must include reachable start, restart, or play controls"
    if "addeventlistener" not in lower and "onclick=" not in lower:
        return "game artifact harness: controls must have a real event handler"
    if not re.search(r"\b(?:requestanimationframe|setinterval|settimeout|classlist\.(?:add|remove|toggle)|textcontent\s*=|innerhtml\s*=)", lower):
        return "game artifact harness: interaction must visibly update game state or UI"
    if re.search(r"position\s*:\s*fixed", value, re.I):
        return "game artifact harness: CSS must not use position:fixed because it can escape the 1920x1080 game stage"
    if re.search(r"(?:width|min-width|max-width)\s*:\s*(?:19[3-9]\d|[2-9]\d{3,})px", value, re.I):
        return "game artifact harness: CSS has an element wider than the 1920px stage"
    if re.search(r"(?:height|min-height|max-height)\s*:\s*(?:10[9]\d|1[1-9]\d{2}|[2-9]\d{3,})px", value, re.I):
        return "game artifact harness: CSS has an element taller than the 1080px stage"
    if not re.search(r"(?<!\d)1920(?!\d)", value) or not re.search(r"(?<!\d)1080(?!\d)", value):
        return "game artifact harness: HTML must define an exact 1920x1080 logical stage"
    if not re.search(r"transform\s*:\s*scale|scale\s*\(", value, re.I):
        return "game artifact harness: HTML must include scale-to-fit logic for the 1920x1080 stage"
    if re.search(r"\b(?:todo|fixme|placeholder)\b", value, re.I):
        return "game artifact harness: remove TODO, FIXME, and placeholder game content before upload"
    return ""


def _game_browserless_harness_source() -> str:
    """Return the Browserless function used to exercise a generated game."""
    return r'''module.exports = async ({ page, context }) => {
  const html = String(context.html || "");
  const timeout = Math.max(1000, Number(context.timeout) || 30000);
  const assetOrigin = "https://alphart-game.local";
  const assets = context.assets && typeof context.assets === "object" ? context.assets : {};
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' ${assetOrigin}; style-src 'unsafe-inline' ${assetOrigin}; img-src data: blob: ${assetOrigin}; media-src data: blob: ${assetOrigin}; font-src data: ${assetOrigin}; connect-src 'none';">`;
  const base = `<base href="${assetOrigin}/">`;
  const sandboxedHTML = /<head\b[^>]*>/i.test(html)
    ? html.replace(/<head\b[^>]*>/i, (head) => head + base + csp)
    : '<!doctype html><head>' + base + csp + '</head>' + html;
  const viewports = [
    { name: "stage", width: 1920, height: 1080 },
    { name: "laptop", width: 1366, height: 768 },
    { name: "tablet", width: 1024, height: 768 },
    { name: "mobile", width: 390, height: 844 },
  ];
  const violations = [];
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(String(error && error.message || error)));
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const url = String(request.url() || "");
    try {
      const parsed = new URL(url);
      if (parsed.origin === assetOrigin) {
        const key = decodeURIComponent(parsed.pathname).replace(/^\\/+/, "");
        const asset = assets[key];
        if (asset && typeof asset.body === "string") {
          request.respond({
            status: 200,
            contentType: String(asset.content_type || "application/octet-stream"),
            headers: { "Access-Control-Allow-Origin": "*" },
            body: Buffer.from(asset.body, "base64"),
          }).catch(() => {});
          return;
        }
      }
    } catch (_) {}
    const allowLocal = /^(?:about:|data:|blob:)/i.test(url);
    const action = allowLocal ? request.continue() : request.abort("blockedbyclient");
    action.catch(() => {});
  });

  for (const viewport of viewports) {
    try {
      await page.setViewport({ width: viewport.width, height: viewport.height, deviceScaleFactor: 1 });
      await page.setContent(sandboxedHTML, { waitUntil: "domcontentloaded", timeout });
      await new Promise((resolve) => setTimeout(resolve, 350));
      const interaction = await page.evaluate(async () => {
        const visible = (element) => {
          if (!element) return false;
          const rect = element.getBoundingClientRect();
          const style = window.getComputedStyle(element);
          return rect.width > 1 && rect.height > 1 && style.display !== "none" && style.visibility !== "hidden";
        };
        const signature = () => {
          const root = document.querySelector("main, [data-game-stage], #stage, #game, canvas, svg");
          const text = `${root ? root.innerHTML : document.body.innerHTML}`;
          let hash = 5381;
          for (let index = 0; index < text.length; index += 1) hash = ((hash * 33) ^ text.charCodeAt(index)) >>> 0;
          const canvasHash = Array.from(document.querySelectorAll("canvas")).map((canvas) => {
            try {
              const context = canvas.getContext("2d");
              if (!context) return `${canvas.width}x${canvas.height}:non-2d`;
              const width = Math.min(canvas.width, 16);
              const height = Math.min(canvas.height, 16);
              const pixels = context.getImageData(0, 0, width, height).data;
              let pixelHash = 5381;
              for (const pixel of pixels) pixelHash = ((pixelHash * 33) ^ pixel) >>> 0;
              return `${canvas.width}x${canvas.height}:${pixelHash}`;
            } catch (_) {
              return `${canvas.width}x${canvas.height}:unreadable`;
            }
          }).join("|");
          return `${hash}:${canvasHash}`;
        };
        const control = Array.from(document.querySelectorAll("button,input[type=button],input[type=submit]")).find((element) => visible(element) && !element.disabled);
        const before = signature();
        const hook = window.__ALPHART_GAME_TEST__;
        if (typeof hook === "function") {
          await hook();
          return { invoked: true, before };
        }
        if (hook && typeof hook.start === "function") {
          await hook.start();
          return { invoked: true, before };
        }
        if (control) {
          control.click();
          return { invoked: true, before };
        }
        return { invoked: false, before };
      });
      await new Promise((resolve) => setTimeout(resolve, 250));
      const audit = await page.evaluate(() => {
        const visible = (element) => {
          if (!element) return false;
          const rect = element.getBoundingClientRect();
          const style = window.getComputedStyle(element);
          return rect.width > 1 && rect.height > 1 && style.display !== "none" && style.visibility !== "hidden";
        };
        const playfields = Array.from(document.querySelectorAll("canvas,svg")).filter(visible);
        const controls = Array.from(document.querySelectorAll("button,input[type=button],input[type=submit]")).filter(visible);
        const root = document.querySelector("[data-game-stage], #stage, #game, main, canvas, svg");
        const visibleGameElements = root
          ? [root, ...Array.from(root.querySelectorAll("*")).filter(visible)]
          : [];
        const rootRect = root ? root.getBoundingClientRect() : null;
        const doc = document.documentElement;
        const body = document.body;
        const outsideViewport = (element) => {
          const rect = element.getBoundingClientRect();
          return rect.left < -1 || rect.top < -1 || rect.right > window.innerWidth + 1 || rect.bottom > window.innerHeight + 1;
        };
        return {
          playfieldCount: playfields.length,
          controlCount: controls.length,
          hasVisibleRoot: visible(root),
          rootOutsideViewport: !!rootRect && (rootRect.right < 0 || rootRect.bottom < 0 || rootRect.left > window.innerWidth || rootRect.top > window.innerHeight),
          rootClippedByViewport: !!rootRect && outsideViewport(root),
          clippedGameElementCount: visibleGameElements.filter((element) => outsideViewport(element)).length,
          outsideStageElementCount: !!rootRect
            ? visibleGameElements.filter((element) => {
                if (element === root) return false;
                const rect = element.getBoundingClientRect();
                return rect.left < rootRect.left - 1 || rect.top < rootRect.top - 1 || rect.right > rootRect.right + 1 || rect.bottom > rootRect.bottom + 1;
              }).length
            : 0,
          horizontalOverflow: Math.max(doc.scrollWidth, body ? body.scrollWidth : 0) > window.innerWidth + 1,
          verticalOverflow: Math.max(doc.scrollHeight, body ? body.scrollHeight : 0) > window.innerHeight + 1,
        };
      });
      const stateChanged = await page.evaluate((before) => {
        const root = document.querySelector("main, [data-game-stage], #stage, #game, canvas, svg");
        const text = `${root ? root.innerHTML : document.body.innerHTML}`;
        let hash = 5381;
        for (let index = 0; index < text.length; index += 1) hash = ((hash * 33) ^ text.charCodeAt(index)) >>> 0;
        const canvasHash = Array.from(document.querySelectorAll("canvas")).map((canvas) => {
          try {
            const context = canvas.getContext("2d");
            if (!context) return `${canvas.width}x${canvas.height}:non-2d`;
            const width = Math.min(canvas.width, 16);
            const height = Math.min(canvas.height, 16);
            const pixels = context.getImageData(0, 0, width, height).data;
            let pixelHash = 5381;
            for (const pixel of pixels) pixelHash = ((pixelHash * 33) ^ pixel) >>> 0;
            return `${canvas.width}x${canvas.height}:${pixelHash}`;
          } catch (_) {
            return `${canvas.width}x${canvas.height}:unreadable`;
          }
        }).join("|");
        return `${hash}:${canvasHash}` !== before;
      }, interaction.before);
      if (!audit.hasVisibleRoot) violations.push(`${viewport.name}: no visible game stage`);
      if (!audit.playfieldCount) violations.push(`${viewport.name}: no visible Canvas or SVG playfield`);
      if (!audit.controlCount) violations.push(`${viewport.name}: no reachable controls`);
      if (!interaction.invoked) violations.push(`${viewport.name}: no start, restart, or test control could be invoked`);
      if (!stateChanged) violations.push(`${viewport.name}: invoking the game control did not change game state`);
      if (audit.rootOutsideViewport) violations.push(`${viewport.name}: game stage is outside the viewport`);
      if (audit.rootClippedByViewport) violations.push(`${viewport.name}: game stage is clipped by the viewport`);
      if (audit.clippedGameElementCount) violations.push(`${viewport.name}: visible game controls or playfield are clipped by the viewport`);
      if (audit.outsideStageElementCount) violations.push(`${viewport.name}: visible game controls or playfield exceed the game stage bounds`);
      if (audit.horizontalOverflow || audit.verticalOverflow) violations.push(`${viewport.name}: document scroll/overflow detected`);
    } catch (error) {
      violations.push(`${viewport.name}: ${String(error && error.message || error)}`);
    }
  }
  for (const error of pageErrors) violations.push(`runtime: ${error}`);
  return { ok: violations.length === 0, violations };
};'''


def _game_runtime_asset_payload(files: Iterable[Tuple[str, bytes, str]]) -> Dict[str, Dict[str, str]]:
    return {
        _safe_game_rel_path(path): {
            "body": base64.b64encode(body).decode("ascii"),
            "content_type": content_type,
        }
        for path, body, content_type in files
        if _safe_game_rel_path(path) != "index.html"
    }


def _game_runtime_harness_feedback(
    html: str, assets: Optional[Dict[str, Dict[str, str]]] = None
) -> str:
    """Run the optional Browserless render harness before a game is uploaded."""
    browserless_url = _game_browserless_url()
    if not browserless_url:
        return ""

    endpoint = f"{browserless_url}/function"
    token = _game_browserless_token()
    if token:
        endpoint = f"{endpoint}?token={quote(token, safe='')}"
    timeout = min(max(_backend_tool_timeout(45), 5), 60)
    try:
        response = requests.post(
            endpoint,
            json={
                "code": _game_browserless_harness_source(),
                "context": {"html": html, "assets": assets or {}, "timeout": timeout * 1000},
            },
            timeout=timeout + 5,
        )
    except requests.RequestException as exc:
        return f"game runtime harness: Browserless request failed: {exc}"
    if response.status_code < 200 or response.status_code >= 300:
        return f"game runtime harness: Browserless returned HTTP {response.status_code}: {response.text[:300]}"
    try:
        report = response.json()
    except ValueError:
        return "game runtime harness: Browserless returned an invalid response"
    if isinstance(report, dict) and isinstance(report.get("data"), dict):
        report = report["data"]
    if not isinstance(report, dict):
        return "game runtime harness: Browserless returned an invalid report"
    if report.get("ok") is True:
        return ""
    violations = report.get("violations")
    if isinstance(violations, list):
        detail = "; ".join(str(item).strip() for item in violations if str(item).strip())
    else:
        detail = str(report.get("error") or report.get("message") or "runtime validation failed").strip()
    return f"game runtime harness: {detail[:800] or 'runtime validation failed'}"


def _prepare_game_html_for_upload(html: str) -> str:
    value = str(html or "")
    if "alphart-game-fit-stage" in value or "__ALPHART_GAME_FIT__" in value:
        return value

    style = """<style id="alphart-game-fit-style">
html,body{margin:0!important;width:100%!important;height:100%!important;overflow:hidden!important;}
body{background:#0b1020;display:block!important;}
*,*::before,*::after{box-sizing:border-box;}
#alphart-game-fit-stage{position:absolute;inset:0;overflow:hidden;background:inherit;}
#alphart-game-fit-content{position:absolute;left:50%;top:50%;width:1920px!important;height:1080px!important;transform-origin:center center;overflow:hidden;max-width:none!important;max-height:none!important;will-change:transform;}
#alphart-game-fit-content > #stage,#alphart-game-fit-content > #game,#alphart-game-fit-content > [data-game-stage]{margin:0!important;left:0!important;top:0!important;transform:none!important;}
</style>"""
    script = """<script id="alphart-game-fit-script">
(function(){
  if(window.__ALPHART_GAME_FIT__)return;window.__ALPHART_GAME_FIT__=true;
  function ready(fn){if(document.readyState==="loading"){document.addEventListener("DOMContentLoaded",fn,{once:true});}else{fn();}}
  ready(function(){
    if(document.getElementById("alphart-game-fit-stage"))return;
    var stage=document.createElement("div");stage.id="alphart-game-fit-stage";
    var content=document.createElement("div");content.id="alphart-game-fit-content";
    var skip={SCRIPT:1,STYLE:1,LINK:1,META:1,TITLE:1};
    Array.prototype.slice.call(document.body.children).forEach(function(el){
      if(skip[el.tagName]||el.id==="alphart-game-fit-stage")return;
      content.appendChild(el);
    });
    document.body.insertBefore(stage,document.body.firstChild);
    stage.appendChild(content);
    var observer=new MutationObserver(function(records){
      records.forEach(function(record){
        Array.prototype.slice.call(record.addedNodes).forEach(function(node){
          if(node.nodeType!==1)return;
          var el=node;
          if(el===stage||el===content||skip[el.tagName]||el.id==="alphart-game-fit-stage")return;
          if(el.parentNode===document.body)content.appendChild(el);
        });
      });
    });
    observer.observe(document.body,{childList:true});
    function layout(){
      var logicalW=1920;
      var logicalH=1080;
      content.style.width=logicalW+"px";
      content.style.height=logicalH+"px";
      var scale=Math.min(window.innerWidth/logicalW,window.innerHeight/logicalH);
      if(!isFinite(scale)||scale<=0)scale=1;
      content.style.transform="translate(-50%,-50%) scale("+scale+")";
    }
    window.addEventListener("resize",layout);
    if("ResizeObserver" in window)new ResizeObserver(layout).observe(content);
    requestAnimationFrame(layout);
    setTimeout(layout,250);
    setTimeout(layout,1000);
  });
})();
</script>"""

    if re.search(r"</head\s*>", value, re.I):
        value = re.sub(r"</head\s*>", style + "\n</head>", value, count=1, flags=re.I)
    elif re.search(r"<body\b", value, re.I):
        value = re.sub(r"<body\b", style + "\n<body", value, count=1, flags=re.I)
    else:
        value = style + "\n" + value

    if re.search(r"</body\s*>", value, re.I):
        return re.sub(r"</body\s*>", script + "\n</body>", value, count=1, flags=re.I)
    return value + "\n" + script


def _strip_invisible_game_html(body: str) -> str:
    value = re.sub(
        r"<!--.*?-->|<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<template\b[^>]*>.*?</template>|<noscript\b[^>]*>.*?</noscript>|<meta\b[^>]*>|<link\b[^>]*>",
        " ",
        body,
        flags=re.I | re.S,
    )
    return _strip_hidden_game_html(value)


def _strip_hidden_game_html(body: str) -> str:
    current = body
    previous = None
    while current != previous:
        previous = current
        current = re.sub(
            r"<([a-z][a-z0-9:-]*)\b[^>]*(?:\shidden(?:\s|=|>)|style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^\"']*[\"'])[^>]*>.*?</\1>",
            " ",
            current,
            flags=re.I | re.S,
        )
        current = re.sub(
            r"<[a-z][a-z0-9:-]*\b[^>]*(?:\shidden(?:\s|=|/?>)|style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^\"']*[\"'])[^>]*/?>",
            " ",
            current,
            flags=re.I | re.S,
        )
    return current


def _request_game_upload_target(args: Dict[str, Any]) -> Dict[str, Any]:
    backend_url = _backend_url()
    if not backend_url:
        raise RuntimeError("ALPHART_EDU_BACKEND_URL is not configured")
    token = _auth_token()
    service_token = _service_token()
    payload = {
        "session_id": args.get("session_id") or _ctx().get("session_id"),
        "canvas_id": args.get("canvas_id") or _ctx().get("canvas_id"),
        "user_uuid": args.get("user_uuid") or _ctx().get("user_uuid"),
        "storage_prefix": args.get("storage_prefix") or _ctx().get("storage_prefix"),
        "org_no": args.get("org_no") or _ctx().get("org_no") or _ctx().get("storage_prefix"),
        "game_id": args.get("game_id"),
        "content_type": "text/html; charset=utf-8",
    }
    resp = requests.post(
        f"{backend_url}/internal/api/v1/agent/game-upload-target",
        json=payload,
        headers={
            **({"Authorization": f"Bearer {service_token or token}"} if (service_token or token) else {}),
            **({"X-Hermes-Agent-Token": service_token} if service_token else {}),
        },
        timeout=_backend_tool_timeout(),
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"game upload target failed {resp.status_code}: {resp.text[:300]}")
    decoded = resp.json()
    if not isinstance(decoded, dict) or not decoded.get("url"):
        raise RuntimeError("game upload target response is invalid")
    decoded.setdefault("upload_url", decoded.get("url"))
    return decoded


def _game_s3_client(target: Dict[str, Any]) -> Tuple[Any, str, str]:
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for game artifact upload; rebuild the agent image with the bedrock extra") from exc

    bucket = str(
        os.getenv("ALPHART_EDU_S3_PUBLIC_BUCKET")
        or os.getenv("S3_PUBLIC_BUCKET")
        or os.getenv("S3_BUCKET")
        or target.get("bucket_name")
        or ""
    ).strip()
    key = str(target.get("s3_object_name") or "").strip()
    endpoint = str(os.getenv("ALPHART_EDU_S3_ENDPOINT") or os.getenv("S3_ENDPOINT") or "").strip()
    region = str(os.getenv("ALPHART_EDU_S3_REGION") or os.getenv("S3_REGION") or "us-east-1").strip()
    access_key = str(os.getenv("ALPHART_EDU_S3_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID") or "").strip()
    secret_key = str(os.getenv("ALPHART_EDU_S3_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY") or "").strip()
    force_path_style = str(os.getenv("ALPHART_EDU_S3_FORCE_PATH_STYLE") or os.getenv("S3_FORCE_PATH_STYLE") or "true").strip().lower() in {"1", "true", "yes", "on"}
    if not bucket or not key:
        raise RuntimeError("game upload target missing bucket or s3_object_name")
    if not access_key or not secret_key:
        raise RuntimeError("ALPHART_EDU_S3_ACCESS_KEY_ID and ALPHART_EDU_S3_SECRET_ACCESS_KEY are required for game upload")
    client_kwargs: Dict[str, Any] = {
        "region_name": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if endpoint:
        if not endpoint.startswith(("http://", "https://")):
            endpoint = "https://" + endpoint
        client_kwargs["endpoint_url"] = endpoint
    if force_path_style:
        from botocore.config import Config  # type: ignore

        client_kwargs["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client("s3", **client_kwargs), bucket, key


def _guess_content_type(path: str, fallback: str = "application/octet-stream") -> str:
    if path.lower().endswith(".js"):
        return "text/javascript; charset=utf-8"
    if path.lower().endswith(".css"):
        return "text/css; charset=utf-8"
    if path.lower().endswith((".html", ".htm")):
        return "text/html; charset=utf-8"
    guessed, _ = mimetypes.guess_type(path)
    return guessed or fallback


def _upload_game_object(target: Dict[str, Any], key: str, body: bytes, content_type: str) -> None:
    s3, bucket, _ = _game_s3_client(target)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        ACL="public-read",
    )


def _upload_game_html(target: Dict[str, Any], html: str) -> None:
    s3, bucket, key = _game_s3_client(target)
    html = _prepare_game_html_for_upload(html)
    feedback = _game_artifact_harness_feedback(html)
    if feedback:
        raise RuntimeError(feedback)
    feedback = _game_runtime_harness_feedback(html)
    if feedback:
        raise RuntimeError(feedback)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=html.encode("utf-8"),
        ContentType=str(target.get("content_type") or "text/html; charset=utf-8"),
        ACL="public-read",
    )


def _safe_game_rel_path(value: Any) -> str:
    rel = str(value or "").strip().replace("\\", "/").lstrip("/")
    parts = [part for part in rel.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise RuntimeError(f"invalid game artifact path: {value!r}")
    return "/".join(parts)


def _game_object_prefix(target: Dict[str, Any]) -> str:
    key = str(target.get("s3_object_name") or "").strip()
    if not key or "/" not in key:
        raise RuntimeError("game upload target missing index.html s3_object_name")
    return key.rsplit("/", 1)[0]


def _upload_game_directory(target: Dict[str, Any], artifact_dir: Path) -> str:
    root = artifact_dir.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"game artifact directory does not exist: {artifact_dir}")
    index_path = root / "index.html"
    if not index_path.is_file():
        raise RuntimeError("game artifact directory must contain index.html")
    html = _prepare_game_html_for_upload(index_path.read_text(encoding="utf-8"))
    feedback = _game_artifact_harness_feedback(html)
    if feedback:
        raise RuntimeError(feedback)
    artifact_files: List[Tuple[str, bytes, str]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        rel = resolved.relative_to(root).as_posix()
        body = resolved.read_bytes()
        artifact_files.append((rel, body, _guess_content_type(rel)))
    runtime_feedback = _game_runtime_harness_feedback(
        html, _game_runtime_asset_payload(artifact_files)
    )
    if runtime_feedback:
        raise RuntimeError(runtime_feedback)

    s3, bucket, index_key = _game_s3_client(target)
    prefix = _game_object_prefix(target)
    uploaded = 0
    for rel, original_body, content_type in artifact_files:
        key = f"{prefix}/{_safe_game_rel_path(rel)}"
        body = html.encode("utf-8") if rel == "index.html" else original_body
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            ACL="public-read",
        )
        uploaded += 1
    if uploaded == 0:
        raise RuntimeError("game artifact directory has no files to upload")
    return index_key


def _file_payload_bytes(item: Dict[str, Any]) -> bytes:
    if item.get("base64"):
        return base64.b64decode(str(item.get("base64") or ""), validate=True)
    content = item.get("content")
    if isinstance(content, bytes):
        return content
    return str(content or "").encode("utf-8")


def _upload_game_files(target: Dict[str, Any], files: List[Any]) -> str:
    if not files:
        raise RuntimeError("game files list is empty")
    prefix = _game_object_prefix(target)
    index_html = ""
    normalized: List[Tuple[str, bytes, str]] = []
    for raw in files:
        if not isinstance(raw, dict):
            raise RuntimeError("game files entries must be objects")
        rel = _safe_game_rel_path(raw.get("path") or raw.get("name") or raw.get("filename"))
        body = _file_payload_bytes(raw)
        content_type = str(raw.get("content_type") or raw.get("mime_type") or _guess_content_type(rel)).strip()
        normalized.append((rel, body, content_type))
        if rel == "index.html":
            index_html = body.decode("utf-8", errors="replace")
    if not index_html:
        raise RuntimeError("game files list must contain index.html")
    prepared_index_html = _prepare_game_html_for_upload(index_html)
    feedback = _game_artifact_harness_feedback(prepared_index_html)
    if feedback:
        raise RuntimeError(feedback)
    runtime_feedback = _game_runtime_harness_feedback(
        prepared_index_html, _game_runtime_asset_payload(normalized)
    )
    if runtime_feedback:
        raise RuntimeError(runtime_feedback)
    s3, bucket, _ = _game_s3_client(target)
    for rel, body, content_type in normalized:
        if rel == "index.html":
            body = prepared_index_html.encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{rel}",
            Body=body,
            ContentType=content_type,
            ACL="public-read",
        )
    return str(target.get("s3_object_name") or "")


def _handle_alphart_generate_game(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    args.setdefault("game_plan", _default_game_plan())
    args.setdefault("layout_requirements", _default_game_layout_requirements())
    args.setdefault("review_checklist", _default_game_review_checklist())
    html = str(args.get("html") or args.get("index_html") or "").strip()
    artifact_path = str(args.get("artifact_dir") or args.get("artifact_path") or args.get("directory") or "").strip()
    files = args.get("files")
    has_files = isinstance(files, list) and len(files) > 0
    if not html and not artifact_path and not has_files:
        return _tool_error("game tool requires html, artifact_dir/artifact_path, or files")
    try:
        target = _request_game_upload_target(args)
        if artifact_path:
            path = Path(artifact_path).expanduser()
            if path.is_dir():
                _upload_game_directory(target, path)
            elif path.is_file():
                html = path.read_text(encoding="utf-8")
                _upload_game_html(target, html)
            else:
                return _tool_error(f"game artifact path does not exist: {artifact_path}")
        elif has_files:
            _upload_game_files(target, files)
        else:
            _upload_game_html(target, html)
    except Exception as exc:
        code = "GAME_VALIDATION_ERROR" if str(exc).lower().startswith(("game artifact harness:", "game runtime harness:")) else ""
        return _tool_error(str(exc), code=code)
    result = {
        "status": "success",
        "result": {
            "type": "generate_game_result",
            "phase": "result",
            "game_id": target.get("game_id") or args.get("game_id"),
            "url": target.get("url"),
            "s3_object_name": target.get("s3_object_name"),
            "summary": args.get("prompt") or args.get("description") or "Generated game",
            "width": 1920,
            "height": 1080,
        },
    }
    return json.dumps(result, ensure_ascii=False)


def _handle_alphart_transcribe_audio(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    audio_url = str(args.get("audio_url") or args.get("url") or "").strip()
    if not audio_url:
        return _tool_error("audio_url is required")
    try:
        from tools.transcription_tools import transcribe_audio  # type: ignore

        suffix = Path(urlparse(audio_url).path).suffix or ".audio"
        with requests.get(audio_url, stream=True, timeout=120) as resp:
            if resp.status_code < 200 or resp.status_code >= 300:
                return _tool_error(f"audio download failed: HTTP {resp.status_code}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
        try:
            result = transcribe_audio(tmp_path, model=str(args.get("model") or "") or None)
        finally:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
    except Exception as exc:
        return _tool_error(f"audio transcription failed: {exc}")
    text = ""
    if isinstance(result, dict):
        text = str(result.get("text") or result.get("transcript") or "").strip()
    else:
        text = str(result or "").strip()
    return json.dumps({"status": "success", "result": {"type": "transcription", "text": text, "raw": result}}, ensure_ascii=False)


WRITE_PLAN_SCHEMA = {
    "name": "write_plan",
    "description": "Create a concise plan before running media generation.",
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["title"],
                },
            }
        },
        "required": ["steps"],
    },
}

CANVAS_CREATE_NODE_SCHEMA = {
    "name": "canvas_create_node",
    "description": (
        "Create a Canvas node in the current document. Use this before generation when the user asks the agent "
        "to create or manage canvas items, especially to create an image/video prompt node that later generation "
        "tools can update by passing canvas_item_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "canvas_id": {"type": "string", "description": "Canvas document id. Defaults to current canvas."},
            "item_type": {"type": "string", "enum": ["text", "note", "image", "video", "audio", "file", "group"]},
            "title": {"type": "string", "description": "Concise 2-6 word summary of this node's role or content; preserve an explicit user title."},
            "text": {"type": "string", "description": "Text content for text/note nodes."},
            "prompt": {"type": "string", "description": "Prompt content for media generation nodes."},
            "content": {"type": "object", "description": "Full Canvas node content JSON."},
            "generation_config": {"type": "object", "description": "Generation config JSON."},
            "position_x": {"type": "number"},
            "position_y": {"type": "number"},
            "width": {"type": "number"},
            "height": {"type": "number"},
            "z_index": {"type": "integer"},
            "last_run_status": {"type": "string", "enum": ["idle", "running", "completed", "failed"]},
            "source_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Canvas node ids to connect into this new node. For referenced inputs, pass every source id; Canvas persists these lines automatically.",
            },
        },
        "required": ["item_type", "title"],
    },
}

CANVAS_UPDATE_NODE_SCHEMA = {
    "name": "canvas_update_node",
    "description": "Update an existing Canvas node's title, content, prompt, status, layout, or output.",
    "parameters": {
        "type": "object",
        "properties": {
            "canvas_id": {"type": "string", "description": "Canvas document id. Defaults to current canvas."},
            "canvas_item_id": {"type": "string", "description": "Canvas item/node id to update."},
            "item_id": {"type": "string", "description": "Alias for canvas_item_id."},
            "item_type": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "prompt": {"type": "string"},
            "content": {"type": "object"},
            "generation_config": {"type": "object"},
            "position_x": {"type": "number"},
            "position_y": {"type": "number"},
            "width": {"type": "number"},
            "height": {"type": "number"},
            "z_index": {"type": "integer"},
            "last_run_status": {"type": "string", "enum": ["idle", "running", "completed", "failed"]},
            "last_run_error": {"type": "string"},
            "last_output": {"type": "object"},
        },
        "required": ["canvas_item_id"],
    },
}

CANVAS_CONNECT_NODES_SCHEMA = {
    "name": "canvas_connect_nodes",
    "description": "Create a connection between two Canvas nodes in the current document.",
    "parameters": {
        "type": "object",
        "properties": {
            "canvas_id": {"type": "string", "description": "Canvas document id. Defaults to current canvas."},
            "source_item_id": {"type": "string"},
            "target_item_id": {"type": "string"},
            "source_handle": {"type": "string", "default": "out"},
            "target_handle": {"type": "string", "default": "in"},
        },
        "required": ["source_item_id", "target_item_id"],
    },
}

CANVAS_GENERATE_IMAGE_SCHEMA = {
    "name": "canvas_generate_image",
    "description": (
        "Generate or edit images through the selected Canvas image model. "
        "Use this immediately after planning for image tasks. The backend stores results in S3 and updates Canvas."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Detailed professional image prompt."},
            "canvas_item_id": {"type": "string", "description": "Existing Canvas image node id to update instead of creating a duplicate node."},
            "tool_id": {"type": "string", "description": "Selected Canvas tool id, when known."},
            "provider": {"type": "string", "description": "Selected image provider, when known."},
            "model": {"type": "string", "description": "Selected image model, when known."},
            "aspect_ratio": {"type": "string", "description": "1:1, 16:9, 9:16, 4:3, or 3:4."},
            "resolution": {"type": "string", "description": "Canvas image resolution: auto, 1K, 2K, or 4K."},
            "image_quantity": {"type": "integer", "description": "Requested number of images."},
            "input_images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s3_object_name": {"type": "string"},
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "role": {"type": "string", "description": "Reference role from the user's @file instruction, e.g. protagonist or background."},
                        "reference_note": {"type": "string", "description": "Short reference note from the user's @file instruction."},
                    },
                },
                "description": "Reference images extracted from <input_images> XML. Prefer s3_object_name objects; file_id strings are fallback only.",
            },
        },
        "required": ["prompt"],
    },
}

CANVAS_GENERATE_VIDEO_SCHEMA = {
    "name": "canvas_generate_video",
    "description": (
        "Submit a video generation task through the selected Canvas video model. "
        "Use this for text-to-video and image-to-video tasks. Seedance result polling stays in the Go backend."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Detailed cinematic video prompt."},
            "tool_id": {"type": "string", "description": "Selected Canvas tool id, when known."},
            "provider": {"type": "string", "description": "Selected video provider, when known."},
            "model": {"type": "string", "description": "Selected video model, when known."},
            "image_url": {"type": "string", "description": "Reference image URL or file_id."},
            "input_images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s3_object_name": {"type": "string"},
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "role": {"type": "string", "description": "Reference role from the user's @file instruction, e.g. protagonist or background."},
                        "reference_note": {"type": "string", "description": "Short reference note from the user's @file instruction."},
                    },
                },
                "description": "Reference images extracted from <input_images> XML. Prefer s3_object_name objects; file_id strings are fallback only.",
            },
            "input_audio": {
                "type": "array",
                "items": {"type": "object", "properties": {"s3_object_name": {"type": "string"}, "url": {"type": "string"}, "filename": {"type": "string"}, "role": {"type": "string", "enum": ["soundtrack", "background_music", "voiceprint"]}}},
                "description": "Soundtrack audio references. Canvas sends these from connected sound nodes.",
            },
            "duration_seconds": {"type": "integer", "description": "Requested video duration in seconds."},
            "resolution": {"type": "string", "description": "Video resolution, for example 480p, 720p, 1080p."},
            "aspect_ratio": {"type": "string", "description": "Video aspect ratio, for example 16:9 or 9:16."},
            "generate_audio": {"type": "boolean", "description": "Whether the generated video should include audio."},
            "caption_script": {
                "type": "string",
                "description": "Ready-to-speak caption/voiceover script for Canvas. Keep it within the requested duration; do not put it in the visual video prompt.",
            },
            "wait": {"type": "boolean", "default": False},
        },
        "required": ["prompt"],
    },
}

# Voiceover and soundtrack wiring belongs to Canvas. Edu receives the same
# video capability without Canvas-only caption metadata or audio-node inputs.
GENERATE_VIDEO_SCHEMA = {
    "name": "generate_video",
    "description": (
        "Submit a video generation task through the selected Alphart video model. "
        "Use this for text-to-video and image-to-video tasks. Video result polling stays in the Go backend."
    ),
    "parameters": {
        **CANVAS_GENERATE_VIDEO_SCHEMA["parameters"],
        "properties": {
            key: value
            for key, value in CANVAS_GENERATE_VIDEO_SCHEMA["parameters"]["properties"].items()
            if key not in {"caption_script", "input_audio"}
        },
    },
}

CANVAS_GENERATE_AUDIO_SCHEMA = {
    "name": "canvas_generate_audio",
    "description": (
        "Generate spoken audio through the selected Alphart Edu TTS/audio model. "
        "Use this for requests like 'generate an audio...', '生成一段音频...', "
        "'生成一段音频用粤语/广东话...', voiceover, narration, read-aloud, or spoken explanation. "
        "Pass ready-to-speak script text as input, not just the user's raw command."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Ready-to-speak script text for the audio."},
            "prompt": {"type": "string", "description": "Alias for input when needed."},
            "tool_id": {"type": "string", "description": "Selected audio/TTS tool id, when known."},
            "provider": {"type": "string", "description": "Selected audio/TTS provider, when known."},
            "model": {"type": "string", "description": "Selected audio/TTS model, when known."},
            "voice": {"type": "string", "description": "Optional voice id/name."},
            "language_type": {
                "type": "string",
                "enum": ["mandarin", "cantonese", "english"],
                "description": "Requested spoken language/accent: mandarin for 中文, cantonese for 粤语/广东话, english for English.",
            },
            "response_format": {"type": "string", "description": "Optional output format, e.g. wav or mp3."},
            "duration_seconds": {"type": "integer", "description": "Requested Canvas audio duration in seconds (5-15)."},
        },
        "required": ["input"],
    },
}

CANVAS_CREATE_STORYBOOK_SCHEMA = {
    "name": "canvas_create_storybook",
    "description": (
        "Create a Gemini-style educational storybook artifact inside Alphart Edu's chat/canvas workflow. "
        "Use this for requests like 'make a storybook', 'create a flip-book lesson', 'storybook of ...', "
        "绘本, 故事书, 童书, 翻页故事, or Traditional Chinese equivalents. "
        "This tool stores a complete illustrated/read-aloud flipbook structure in the Edu backend and "
        "generates the image pages before returning a compact card payload. Do not mirror external book APIs; "
        "adapt the story/topic/prompt into a canvas-native 10-page storybook experience. A text-only page plan "
        "is not a successful storybook result. Storybook physical pages and generated illustrations must be strict "
        "1:1 square pages for printer compatibility."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the storybook."},
            "template_id": {"type": "integer", "description": "Optional backend template id."},
            "template_slug": {"type": "string", "description": "Optional template slug, e.g. science, hero, two-character."},
            "template_name": {"type": "string", "description": "Human readable template name."},
            "category": {"type": "string", "description": "Template/category label, e.g. science, math, personalized."},
            "topic": {"type": "string", "description": "Learning topic or story premise."},
            "prompt": {"type": "string", "description": "User-facing educational storybook brief."},
            "description": {"type": "string", "description": "Optional one-paragraph description."},
            "language": {"type": "string", "description": "Language code or name, e.g. en, zh-CN, zh-TW."},
            "read_aloud_language": {
                "type": "string",
                "enum": ["mandarin", "english", "cantonese"],
                "description": (
                    "Spoken language for read-aloud audio. Default mandarin for Chinese storybooks and english for English storybooks. "
                    "Use cantonese only when the user explicitly asks 用粤语/粵語/广东话/廣東話朗读/read aloud."
                ),
            },
            "age_range": {"type": "string", "description": "Target learner age range, e.g. 6-9."},
            "reading_level": {"type": "string", "description": "Target reading level."},
            "style": {"type": "string", "description": "Visual style for later page image generation."},
            "page_count": {"type": "integer", "description": "Number of pages, 2 to 16. Default to 10 unless the user asks otherwise."},
            "read_aloud": {"type": "boolean", "description": "Whether to prepare read-aloud narration. Default true."},
            "aspect_ratio": {
                "type": "string",
                "description": "Storybook image aspect ratio. Must be 1:1 for strict square printer pages.",
                "enum": ["1:1"],
            },
            "generate_images": {
                "type": "boolean",
                "description": "Whether the backend must generate storybook page illustrations during this tool call. Default true; do not set false for normal storybook creation.",
            },
            "input_images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s3_object_name": {"type": "string"},
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "role": {"type": "string", "description": "Reference role from the user's @file instruction, e.g. protagonist or background."},
                        "reference_note": {"type": "string", "description": "Short reference note from the user's @file instruction."},
                    },
                },
                "description": (
                    "Attached protagonist/character reference images. Prefer s3_object_name objects; "
                    "do not pass presigned URLs unless no object key exists."
                ),
            },
            "protagonists": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "reference_image": {"type": "string"},
                    },
                },
                "description": "Optional named protagonist notes for maintaining character consistency across pages.",
            },
            "pages": {
                "type": "array",
                "description": (
                    "Agent-authored page plan. Must be a native array, not a JSON string or markdown. "
                    "Include a cover page, alternating image/narration story pages when appropriate, "
                    "and a closing/back-cover page. Image pages must include 1:1 image_prompt. "
                    "Narration pages should include narration/read-aloud text and may omit image_prompt. "
                    "Keep each page object compact; do not include long metadata."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {"type": "integer", "description": "Human page number starting at 1."},
                        "page_index": {"type": "integer", "description": "Zero-based page index."},
                        "page_type": {
                            "type": "string",
                            "description": "cover, image, narration, story, closing, or back-cover.",
                        },
                        "layout": {
                            "type": "string",
                            "description": "cover, flipbook-page, image, narration, back-cover, or another compact layout label.",
                        },
                        "title": {"type": "string"},
                        "narration": {
                            "type": "string",
                            "description": (
                                "Short read-aloud text for this page. Keep age-appropriate and concise. "
                                "Default to Mandarin written Chinese for Chinese storybooks and English for English storybooks; "
                                "write Cantonese narration only when read_aloud_language is cantonese."
                            ),
                        },
                        "image_prompt": {
                            "type": "string",
                            "description": (
                                "Strict 1:1 page illustration prompt. Repeat character anchors, style bible, "
                                "visual evidence for the narration, and no text-in-image unless essential."
                            ),
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional compact metadata only. Omit unless truly needed.",
                        },
                    },
                    "required": ["page_number", "page_type", "title"],
                },
            },
        },
        "required": ["topic"],
    },
}

CANVAS_UPDATE_STORYBOOK_PAGE_SCHEMA = {
    "name": "canvas_update_storybook_page",
    "description": (
        "Revise a specific page in an existing Alphart Edu storybook. Use this when the user says "
        "@page 1 image is wrong, @page 2 replace the protagonist, revise this storybook page, or "
        "equivalent Chinese/Traditional Chinese. Read the latest storybook_id/page list from chat "
        "history and pass page_number or page_index. Preserve s3_object_name reference images."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "storybook_id": {"type": "string", "description": "Existing storybook id from the previous storybook result."},
            "page_id": {"type": "string", "description": "Optional page id."},
            "page_number": {"type": "integer", "description": "Human page number, e.g. 1 for @page 1."},
            "page_index": {"type": "integer", "description": "Zero-based page index."},
            "title": {"type": "string", "description": "Optional replacement page title."},
            "narration": {"type": "string", "description": "Optional replacement read-aloud narration."},
            "image_prompt": {"type": "string", "description": "Optional replacement image prompt for this page."},
            "instructions": {"type": "string", "description": "Concise user requested change for this page."},
            "input_images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "s3_object_name": {"type": "string"},
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "role": {"type": "string", "description": "Reference role from the user's @file instruction, e.g. protagonist or background."},
                        "reference_note": {"type": "string", "description": "Short reference note from the user's @file instruction."},
                    },
                },
                "description": "Reference/protagonist images for the page revision.",
            },
        },
        "required": ["storybook_id"],
    },
}


registry.register(
    name="write_plan",
    toolset="alphart-edu",
    schema=WRITE_PLAN_SCHEMA,
    handler=_handle_write_plan,
    is_async=False,
)
registry.register(
    name="canvas_create_node",
    toolset="alphart-canvas",
    schema=CANVAS_CREATE_NODE_SCHEMA,
    handler=_handle_canvas_create_node,
    is_async=False,
)
registry.register(
    name="canvas_update_node",
    toolset="alphart-canvas",
    schema=CANVAS_UPDATE_NODE_SCHEMA,
    handler=_handle_canvas_update_node,
    is_async=False,
)
registry.register(
    name="canvas_connect_nodes",
    toolset="alphart-canvas",
    schema=CANVAS_CONNECT_NODES_SCHEMA,
    handler=_handle_canvas_connect_nodes,
    is_async=False,
)
registry.register(
    name="canvas_create_storybook",
    toolset="alphart-edu",
    schema=CANVAS_CREATE_STORYBOOK_SCHEMA,
    handler=_handle_alphart_create_storybook,
    is_async=False,
)
registry.register(
    name="create_storybook",
    toolset="alphart-edu",
    schema={**CANVAS_CREATE_STORYBOOK_SCHEMA, "name": "create_storybook"},
    handler=_handle_alphart_create_storybook,
    is_async=False,
)
registry.register(
    name="canvas_update_storybook_page",
    toolset="alphart-edu",
    schema=CANVAS_UPDATE_STORYBOOK_PAGE_SCHEMA,
    handler=_handle_alphart_update_storybook_page,
    is_async=False,
)
registry.register(
    name="update_storybook_page",
    toolset="alphart-edu",
    schema={**CANVAS_UPDATE_STORYBOOK_PAGE_SCHEMA, "name": "update_storybook_page"},
    handler=_handle_alphart_update_storybook_page,
    is_async=False,
)
registry.register(
    name="canvas_generate_image",
    toolset="alphart-edu",
    schema=CANVAS_GENERATE_IMAGE_SCHEMA,
    handler=_handle_alphart_generate_image,
    is_async=False,
)
registry.register(
    name="generate_image",
    toolset="alphart-edu",
    schema={**CANVAS_GENERATE_IMAGE_SCHEMA, "name": "generate_image"},
    handler=_handle_alphart_generate_image,
    is_async=False,
)
registry.register(
    name="canvas_generate_video",
    toolset="alphart-canvas",
    schema=CANVAS_GENERATE_VIDEO_SCHEMA,
    handler=_handle_alphart_generate_video,
    is_async=False,
)
registry.register(
    name="generate_video",
    toolset="alphart-edu",
    schema=GENERATE_VIDEO_SCHEMA,
    handler=_handle_alphart_generate_video,
    is_async=False,
)
registry.register(
    name="canvas_generate_audio",
    toolset="alphart-edu",
    schema=CANVAS_GENERATE_AUDIO_SCHEMA,
    handler=_handle_alphart_generate_audio,
    is_async=False,
)
registry.register(
    name="generate_audio",
    toolset="alphart-edu",
    schema={**CANVAS_GENERATE_AUDIO_SCHEMA, "name": "generate_audio"},
    handler=_handle_alphart_generate_audio,
    is_async=False,
)

CANVAS_GENERATE_GAME_SCHEMA = {
    "name": "canvas_generate_game",
    "description": (
        "Upload a complete self-contained educational HTML game generated by the agent. "
        "Default style should be a simple pixel-art game with a real play loop, not a plain "
        "web form or static quiz. Good patterns include platformer, top-down maze, arcade "
        "matcher, drag-and-drop sorter, physics launcher, simulation sandbox, boss challenge, "
        "and story quest. Because this is for an education platform, preserve content "
        "precision: formulas, units, definitions, terminology, names, dates, and causal "
        "relationships must stay correct. Use this for 'make a game that teaches/explains ...' or "
        "'create a quiz/game about ...' requests. Before calling, create a strict plan internally. "
        "Do not call file-writing/coding tools such as Write, Edit, MultiEdit, Bash, write_file, patch, "
        "terminal, or process; provide the complete self-contained playable document in html "
        "on the first call. Do not call this tool with only a prompt, a plan, or placeholder HTML. "
        "The HTML must implement its own exact 1920x1080 logical stage plus scale-to-fit logic so the public game URL "
        "shows the whole game in a normal browser tab without scrollbars; do not rely on Canvas iframe wrapping. "
        "After the tool returns, review content, UI layout, and interactions; if the result "
        "has clipped text, overflow, overlapping controls, or out-of-frame elements, revise "
        "the prompt and regenerate instead of finalizing. Keep optional planning fields concise; "
        "the tool will apply the default plan, layout, and QA checklist when they are omitted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the game to generate, including the concept/topic to teach and any specific requirements.",
            },
            "template": {
                "type": "string",
                "enum": ["pixel_platformer", "topdown_maze", "arcade_matcher", "drag_sorter", "simulation", "boss_challenge", "physics_launcher", "story_quest", "quiz", "concept"],
                "description": (
                    "Optional game pattern. Prefer pixel_platformer, topdown_maze, arcade_matcher, "
                    "drag_sorter, simulation, boss_challenge, physics_launcher, or story_quest. "
                    "Use quiz/concept only when the user explicitly wants a quiz or pure explainer."
                ),
            },
            "game_id": {"type": "string", "description": "Optional stable identifier for the game."},
            "html": {
                "type": "string",
                "description": (
                    "Complete self-contained playable game HTML. Must start with <!DOCTYPE html>, "
                    "include a <body> with visible game content, inline CSS/JS, and closing </body></html>. "
                    "Prefer a visible game root/stage, HUD or progress, playfield/canvas/SVG, instructions, "
                    "and start/restart/control elements. Use an exact 1920x1080 logical stage that scales as a whole "
                    "to the iframe/browser viewport with no page scroll, clipped controls, or overlapping panels. "
                    "Do not use position:fixed; position every HUD/dialog/control inside the 1920x1080 stage."
                ),
            },
            "game_plan": {
                "type": "object",
                "description": (
                    "Optional concise planning payload: learning goal, audience, precise knowledge points, "
                    "selected game pattern, core loop, player controls, hazards/collectibles/targets, "
                    "rules, screen states, pixel-art style, layout grid/safe area, assets, and "
                    "success/failure states."
                ),
                "properties": {
                    "learning_goal": {"type": "string"},
                    "audience": {"type": "string"},
                    "knowledge_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact facts, formulas, definitions, vocabulary, units, dates, symbols, or source constraints to preserve.",
                    },
                    "content_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source-of-truth fact ledger. Every educational label, answer, hazard, collectible, dialog, and feedback message must trace to one of these facts or stable common knowledge.",
                    },
                    "answer_option_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Question/option groups. Each group must include at least one correct option; single-answer groups must have exactly one correct option.",
                    },
                    "misconceptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Likely wrong ideas and the precise feedback that corrects them.",
                    },
                    "game_pattern": {"type": "string"},
                    "template_reason": {"type": "string"},
                    "core_loop": {"type": "string"},
                    "player_controls": {"type": "array", "items": {"type": "string"}},
                    "hazards": {"type": "array", "items": {"type": "string"}},
                    "collectibles": {"type": "array", "items": {"type": "string"}},
                    "rules": {"type": "array", "items": {"type": "string"}},
                    "screen_states": {"type": "array", "items": {"type": "string"}},
                    "layout_plan": {"type": "string"},
                    "asset_plan": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
            },
            "layout_requirements": {
                "type": "object",
                "description": (
                    "Responsive layout constraints. Must require no overflow, no clipped text, "
                    "no overlap, readable labels, all controls inside the game frame, and every widget/window "
                    "inside the 40px safe-area inset of the 1920x1080 stage or equivalent scaled frame."
                ),
            },
            "review_checklist": {
                "type": "object",
                "description": (
                    "QA checklist used after generation. Include content accuracy, layout/frame "
                    "fit, border overflow, text clipping, interaction states, and mobile/desktop fit."
                ),
            },
        },
        "required": ["prompt", "html"],
    },
}

registry.register(
    name="canvas_generate_game",
    toolset="alphart-edu",
    schema=CANVAS_GENERATE_GAME_SCHEMA,
    handler=_handle_alphart_generate_game,
    is_async=False,
)
registry.register(
    name="generate_game",
    toolset="alphart-edu",
    schema={**CANVAS_GENERATE_GAME_SCHEMA, "name": "generate_game"},
    handler=_handle_alphart_generate_game,
    is_async=False,
)

CANVAS_TRANSCRIBE_AUDIO_SCHEMA = {
    "name": "canvas_transcribe_audio",
    "description": (
        "Transcribe audio to text (STT) through the selected Canvas audio model. "
        "Use this when the user provides an audio URL or asks to transcribe audio. "
        "Returns the transcribed text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_url": {"type": "string", "description": "URL of the audio file to transcribe."},
            "tool_id": {"type": "string", "description": "Selected Canvas tool id, when known."},
            "provider": {"type": "string", "description": "Selected audio provider, when known."},
            "model": {"type": "string", "description": "Selected audio model, when known."},
            "language_type": {
                "type": "string",
                "enum": ["mandarin", "cantonese", "english"],
                "description": "Recording language intent. Use mandarin for 中文介绍, cantonese for 粤语介绍, and english for Explain in English.",
            },
            "language": {"type": "string", "description": "BCP-47 language code hint (e.g. en, zh, ja). Optional."},
        },
        "required": ["audio_url"],
    },
}

registry.register(
    name="canvas_transcribe_audio",
    toolset="alphart-edu",
    schema=CANVAS_TRANSCRIBE_AUDIO_SCHEMA,
    handler=_handle_alphart_transcribe_audio,
    is_async=False,
)
registry.register(
    name="transcribe_audio",
    toolset="alphart-edu",
    schema={**CANVAS_TRANSCRIBE_AUDIO_SCHEMA, "name": "transcribe_audio"},
    handler=_handle_alphart_transcribe_audio,
    is_async=False,
)
