#!/usr/bin/env python3
"""HTTP service wrapper for running Hermes as an Alphart agent."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
import base64
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("alphart_agent")

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI, APIConnectionError as OpenAIConnectionError, APIStatusError as OpenAIStatusError, APITimeoutError as OpenAITimeoutError
from pydantic import BaseModel, Field
import requests

from run_agent import AIAgent
from tools.skills_sync import sync_skills
from tools.alphart_tools import (
    _ctx,
    _handle_alphart_create_storybook,
    _handle_alphart_generate_audio,
    _handle_alphart_generate_image,
    _handle_alphart_generate_video,
    _handle_alphart_transcribe_audio,
    _handle_alphart_update_storybook_page,
    _selected_tools,
    alphart_context,
)


class AlphartEduChatRequest(BaseModel):
    session_id: str = ""
    canvas_id: str = ""
    canvas_item_id: str = ""
    selected_canvas_item_id: str = ""
    selected_canvas_item_type: str = ""
    canvas_item_type: str = ""
    force_media_intent: str = ""
    canvas_prompt_context: str = ""
    image_model: str = ""
    image_aspect_ratio: str = ""
    image_quality: str = ""
    image_resolution: str = ""
    video_model: str = ""
    audio_model: str = ""
    input_images: List[Any] = Field(default_factory=list)
    input_audio: List[Any] = Field(default_factory=list)
    reference_item_ids: List[str] = Field(default_factory=list)
    duration_seconds: int = 0
    audio_duration_seconds: int = 0
    aspect_ratio: str = ""
    resolution: str = ""
    generate_audio: bool = False
    video_caption_script: str = ""
    script_only: bool = False
    approved_audio_script: str = ""
    user_id: str = ""
    user_uuid: str = ""
    storage_prefix: str = ""
    org_no: str = ""
    auth_token: str = ""
    messages: List[Any] = Field(default_factory=list)
    text_model: Dict[str, Any] = Field(default_factory=dict)
    text_models: List[Dict[str, Any]] = Field(default_factory=list)
    multimodal_model: Dict[str, Any] = Field(default_factory=dict)
    tool_list: List[Any] = Field(default_factory=list)
    model_configs: Dict[str, Any] = Field(default_factory=dict)
    backend_url: str = ""
    app_scope: str = "edu"
    system_prompt: str = ""
    audio_language_type: str = ""
    ui_language: str = ""


class AlphartEduTitleRequest(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_id: str = ""
    auth_token: str = ""
    org_no: str = ""
    backend_url: str = ""
    app_scope: str = "edu"
    text_model: Dict[str, Any] = Field(default_factory=dict)
    text_models: List[Dict[str, Any]] = Field(default_factory=list)
    model_configs: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Alphart Hermes Agent", version="1.0.0")
SYSTEM_BUSY_MESSAGE = "System busy, please try again later."
INSUFFICIENT_CREDITS_MESSAGE = "Insufficient credits. Please top up or upgrade your plan."


def _sync_bundled_skills() -> None:
    try:
        result = sync_skills(quiet=True)
        copied = len(result.get("copied") or [])
        updated = len(result.get("updated") or [])
        if copied or updated:
            logger.info("synced bundled skills copied=%s updated=%s", copied, updated)
    except Exception as exc:
        logger.warning("failed to sync bundled skills: %s", exc)


_sync_bundled_skills()


def _service_token() -> str:
    return (
        os.getenv("HERMES_AGENT_TOKEN")
        or os.getenv("ALPHART_AGENT_TOKEN")
        or os.getenv("CANVAS_AGENT_TOKEN")
        or ""
    ).strip()


def _check_auth(authorization: Optional[str]) -> None:
    token = _service_token()
    if not token:
        return
    expected = f"Bearer {token}"
    if (authorization or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid hermes agent token")


def _string(value: Any) -> str:
    return str(value or "").strip()


def _infer_storybook_language(text: str) -> str:
    value = _string(text)
    if re.search(r"[\u4e00-\u9fff]", value):
        if re.search(r"[繪書頁學習兒童臺灣繁體]", value):
            return "zh-TW"
        return "zh-CN"
    return "en"


def _explicit_storybook_cantonese_read_aloud(text: str) -> bool:
    value = _string(text).lower()
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
    normalized = _string(value).strip().lower()
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


def _ui_audio_language_type(ui_language: Any) -> str:
    return "mandarin" if _string(ui_language).lower().startswith("zh") else "english"


def _storybook_read_aloud_language(text: str, language: str = "", requested: str = "") -> str:
    if _explicit_storybook_cantonese_read_aloud(text):
        return "cantonese"
    normalized_requested = _normalize_audio_language_type(requested)
    if normalized_requested:
        return normalized_requested
    lang = _string(language).lower()
    if any(token in lang for token in ("zh", "chinese", "中文")) or re.search(r"[\u4e00-\u9fff]", text or ""):
        return "mandarin"
    return "english"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_insufficient_credits_error(value: Any) -> bool:
    text = _string(value).lower()
    return (
        "insufficient_credits" in text
        or "insufficient credits" in text
        or "http 402" in text
        or "error code: 402" in text
    )


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_TITLE_LABEL_RE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?\s*(?:title|标题|標題)\s*(?:\*\*)?\s*[:：\-]\s*", re.IGNORECASE)
_TITLE_REQUEST_PREFIX_RE = re.compile(
    r"^\s*(?:a\s+|an\s+|the\s+)?(?:image\s+)?request\s+(?:to\s+(?:generate|create|make|draw|produce)\s+|for\s+(?:generating|creating|making|drawing|producing)\s+|for\s+)",
    re.IGNORECASE,
)
_TITLE_COMMAND_PREFIX_RE = re.compile(r"^\s*(?:generate|create|make|draw|produce)\s+(?:an?\s+|the\s+)?", re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def _clean_llm_title(text: str) -> str:
    title = _strip_think_tags(text)
    title = title.strip().strip(" \t\r\n\"'`*")
    title = _TITLE_LABEL_RE.sub("", title).strip(" \t\r\n\"'`*-")
    title = _TITLE_REQUEST_PREFIX_RE.sub("", title).strip(" \t\r\n\"'`*-")
    title = _TITLE_COMMAND_PREFIX_RE.sub("", title).strip(" \t\r\n\"'`*-")
    title = re.sub(r"^(?:a|an|the)\s+", "", title, flags=re.IGNORECASE).strip()
    title = title.splitlines()[0].strip() if title else ""
    if len(title) > 80:
        title = title[:80].rstrip()
    return title


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return _string(message)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(_string(item.get("text")))
            elif item.get("type") == "image_url":
                raw = item.get("image_url")
                if isinstance(raw, dict):
                    raw = raw.get("url")
                if raw:
                    parts.append(f"[image: {raw}]")
        return "\n".join(part for part in parts if part)
    return _string(content)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return message


def _has_media_content(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"image_url", "input_image", "video_url", "input_video", "video"}:
            return True
    return False


def _backend_media_url(req: AlphartEduChatRequest, ref: Dict[str, Any]) -> str:
    raw_url = _string(ref.get("url") or ref.get("uri"))
    if raw_url:
        return raw_url
    object_name = _string(ref.get("s3_object_name") or ref.get("object_name") or ref.get("key"))
    if not object_name:
        return ""
    backend_url = _backend_url_from_req(req)
    file_id = _string(ref.get("file_id"))
    route_file_id = (object_name.rstrip("/").rsplit("/", 1)[-1] or file_id.rstrip("/").rsplit("/", 1)[-1] or "media")
    return f"{backend_url}/api/v1/files/{quote(route_file_id, safe='')}?s3_object_name={quote(object_name, safe='')}"


def _download_image_as_data_url(req: AlphartEduChatRequest, ref: Dict[str, Any]) -> str:
    url = _backend_media_url(req, ref)
    if not url:
        return ""
    if url.startswith("data:image/"):
        return url
    if not url.startswith(("http://", "https://")):
        return ""
    headers: Dict[str, str] = {}
    if req.auth_token:
        headers["Authorization"] = f"Bearer {req.auth_token}"
    service_token = _service_token()
    if service_token:
        headers["X-Hermes-Agent-Token"] = service_token
    try:
        resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[alphart-agent] image content hydrate failed url={url} error={exc}", flush=True)
        return ""
    mime_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip()
    if not mime_type.startswith("image/"):
        mime_type = _string(ref.get("mime_type") or ref.get("mimeType")) or "image/png"
    if not mime_type.startswith("image/"):
        return ""
    return f"data:{mime_type};base64,{base64.b64encode(resp.content).decode('ascii')}"


def _prepare_chat_content_for_model(req: AlphartEduChatRequest, content: Any) -> Any:
    if not isinstance(content, list):
        return content
    prepared: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            prepared.append(item)
            continue
        part_type = item.get("type")
        if part_type in {"image_url", "input_image"}:
            image_ref = item.get("image_url")
            if not isinstance(image_ref, dict):
                image_ref = {"url": _string(image_ref)}
            data_url = _download_image_as_data_url(req, image_ref)
            if data_url:
                prepared.append({"type": "image_url", "image_url": {"url": data_url}})
            continue
        if part_type in {"video_url", "input_video", "video"}:
            raw_ref = item.get("video_url") or item.get("video") or item
            ref = raw_ref if isinstance(raw_ref, dict) else {"url": _string(raw_ref)}
            video_url = _backend_media_url(req, ref)
            if video_url:
                prepared.append({"type": "text", "text": f"Video reference URL: {video_url}"})
            continue
        prepared.append(item)
    return prepared


def _provider_config(req: Any) -> Dict[str, Any]:
    return _provider_config_for(req.model_configs, req.text_model)


def _provider_config_for(model_configs: Any, text_model: Dict[str, Any]) -> Dict[str, Any]:
    return _provider_config_for_domain(model_configs, "text", text_model)


def _provider_config_for_domain(model_configs: Any, domain: str, model_ref: Dict[str, Any]) -> Dict[str, Any]:
    provider = _string(model_ref.get("provider"))
    model = _string(model_ref.get("model"))
    def with_model_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(raw)
        models = raw.get("models")
        if isinstance(models, dict) and model and isinstance(models.get(model), dict):
            merged.update(models[model])
        return merged
    if not isinstance(model_configs, dict):
        return {}
    config = model_configs.get(domain)
    if isinstance(config, dict) and provider:
        raw = config.get(provider)
        if isinstance(raw, dict):
            return with_model_config(raw)
        # Canvas app_configs may be stored as one flat provider configuration,
        # rather than a provider-keyed map. Accept that shape only when it
        # explicitly names the requested provider.
        if _string(config.get("provider")).lower() == provider.lower():
            return with_model_config(config)
    if provider:
        raw = model_configs.get(provider)
        if isinstance(raw, dict):
            return with_model_config(raw)
    return {}


def _text_model_candidates(req: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in req.text_models:
        if isinstance(item, dict) and _string(item.get("provider")) and _string(item.get("model")):
            candidates.append(item)
    if not candidates and isinstance(req.text_model, dict):
        if _string(req.text_model.get("provider")) and _string(req.text_model.get("model")):
            candidates.append(req.text_model)
    return candidates


def _openai_text_model_candidates(req: Any, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        provider = _string(candidate.get("provider"))
        model = _string(candidate.get("model"))
        config = _provider_config_for(req.model_configs, candidate)
        if _text_model_wire_format(provider, model, config) == "openai":
            out.append(candidate)
    return out


def _require_openai_text_model_candidates(
    req: Any,
    candidates: List[Dict[str, Any]],
    purpose: str,
    *,
    exclude_small_models: bool = False,
) -> List[Dict[str, Any]]:
    openai_candidates = _openai_text_model_candidates(req, candidates)
    if not openai_candidates:
        raise HTTPException(status_code=400, detail=f"{purpose} requires an active OpenAI/GPT text model")
    if exclude_small_models:
        strong_candidates = [
            candidate
            for candidate in openai_candidates
            if not re.search(r"(?:^|[-_.:/])(?:mini|nano|small|lite)(?:$|[-_.:/])", _string(candidate.get("model")).lower())
        ]
        if not strong_candidates:
            raise HTTPException(status_code=400, detail=f"{purpose} requires a non-mini OpenAI/GPT text model")
        if len(strong_candidates) != len(openai_candidates):
            skipped_small = [
                f"{_string(item.get('provider'))}/{_string(item.get('model'))}"
                for item in openai_candidates
                if item not in strong_candidates
            ]
            print(
                f"[alphart-agent] {purpose} skipping mini/small text models; skipped={','.join(skipped_small)}",
                flush=True,
            )
        openai_candidates = strong_candidates
    if len(openai_candidates) != len(candidates):
        skipped = [
            f"{_string(item.get('provider'))}/{_string(item.get('model'))}"
            for item in candidates
            if item not in openai_candidates
        ]
        print(
            f"[alphart-agent] {purpose} using OpenAI/GPT text model only; skipped={','.join(skipped)}",
            flush=True,
        )
    return openai_candidates


def _api_key(config: Dict[str, Any]) -> str:
    for key in ("api_key", "apiKey", "api key", "key"):
        value = _string(config.get(key))
        if value:
            return value
    return ""


def _endpoint(config: Dict[str, Any]) -> str:
    for key in ("endpoint", "api_url", "url", "base_url"):
        value = _string(config.get(key))
        if value:
            return value
    return ""


def _request_app_scope(req: Any) -> str:
    raw = _string(
        getattr(req, "app_scope", "")
        or getattr(req, "app_name", "")
        or getattr(req, "app", "")
    ).lower()
    if "canvas" in raw:
        return "canvas"
    if "edu" in raw:
        return "edu"

    backend_url = _string(getattr(req, "backend_url", "")).rstrip("/")
    canvas_url = _string(os.getenv("ALPHART_CANVAS_BACKEND_URL") or os.getenv("CANVAS_BACKEND_URL")).rstrip("/")
    edu_url = _string(os.getenv("ALPHART_EDU_BACKEND_URL")).rstrip("/")
    if backend_url and canvas_url and backend_url == canvas_url:
        return "canvas"
    if backend_url and edu_url and backend_url == edu_url:
        return "edu"
    if "canvas" in backend_url.lower():
        return "canvas"
    return "edu"


def _canvas_reasoning_config(req: Any) -> Optional[Dict[str, Any]]:
    """Translate Canvas' text think-level setting for Hermes only.

    Edu requests intentionally keep Hermes' existing reasoning behavior. The
    Canvas UI sends the setting alongside the selected text model so it can be
    applied without adding another app-wide environment variable.
    """
    if _request_app_scope(req) != "canvas":
        return None
    text_model = getattr(req, "text_model", {})
    if not isinstance(text_model, dict):
        return None
    effort = _string(
        text_model.get("thinking_level")
        or text_model.get("thinkingLevel")
        or text_model.get("reasoning_effort")
    )
    if not effort:
        return None
    from hermes_constants import parse_reasoning_effort
    return parse_reasoning_effort(effort)


def _backend_url_from_req(req: Any) -> str:
    explicit = _string(getattr(req, "backend_url", "")).rstrip("/")
    if _request_app_scope(req) == "canvas":
        return _string(
            os.getenv("ALPHART_CANVAS_BACKEND_URL")
            or os.getenv("CANVAS_BACKEND_URL")
            or explicit
            or "http://localhost:9999"
        ).rstrip("/")
    return _string(os.getenv("ALPHART_EDU_BACKEND_URL") or explicit or "http://localhost:57988").rstrip("/")


def _internal_relay_base_url(req: Any) -> str:
    return _backend_url_from_req(req) + "/internal"


def _internal_relay_gemini_base_url(req: Any) -> str:
    return _backend_url_from_req(req) + "/internal/gemini/v1beta"


def _internal_relay_headers(req: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    service_token = _service_token()
    if service_token:
        headers["X-Hermes-Agent-Token"] = service_token
    user_id = _string(getattr(req, "user_id", ""))
    if user_id:
        headers["X-Internal-User-ID"] = user_id
    user_uuid = _string(getattr(req, "user_uuid", ""))
    if user_uuid:
        headers["X-Internal-User-UUID"] = user_uuid
    session_id = _string(getattr(req, "session_id", ""))
    if session_id:
        headers["X-Session-ID"] = session_id
    canvas_id = _string(getattr(req, "canvas_id", ""))
    if canvas_id:
        headers["X-Canvas-ID"] = canvas_id
    org_no = _string(getattr(req, "org_no", "")) or _string(getattr(req, "storage_prefix", ""))
    if org_no:
        headers["X-Org-No"] = org_no
    return headers


def _use_internal_relay(req: Any) -> bool:
    return bool(_string(getattr(req, "user_id", "")))


def _internal_relay_api_key() -> str:
    return _service_token() or "internal-relay"


def _merge_extra_headers(kwargs: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    merged = dict(kwargs or {})
    existing = merged.get("extra_headers")
    extra: Dict[str, str] = {}
    if isinstance(existing, dict):
        extra.update({str(k): str(v) for k, v in existing.items()})
    extra.update({str(k): str(v) for k, v in (headers or {}).items() if str(v)})
    if extra:
        merged["extra_headers"] = extra
    return merged


class _InternalRelayAnthropicMessages:
    """Tiny wrapper around the official Anthropic SDK messages resource.

    The SDK owns request/response parsing. This wrapper only injects Alphart's
    internal relay headers because Anthropic-mode Hermes does not pass
    request_overrides.extra_headers into messages.create/stream.
    """

    def __init__(self, messages: Any, headers: Dict[str, str]):
        self._messages = messages
        self._headers = dict(headers or {})

    def create(self, **kwargs: Any) -> Any:
        return self._messages.create(**_merge_extra_headers(kwargs, self._headers))

    def stream(self, **kwargs: Any) -> Any:
        return self._messages.stream(**_merge_extra_headers(kwargs, self._headers))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _InternalRelayAnthropicClient:
    """Derived client facade over the official Anthropic SDK client."""

    def __init__(self, client: Any, headers: Dict[str, str]):
        self._client = client
        self.messages = _InternalRelayAnthropicMessages(client.messages, headers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _install_internal_relay_anthropic_headers(agent: Any, headers: Dict[str, str]) -> None:
    if not headers or getattr(agent, "_anthropic_client", None) is None:
        return
    agent._anthropic_client = _InternalRelayAnthropicClient(agent._anthropic_client, headers)


def _selected_tool_lines(tools: List[Any]) -> List[str]:
    lines: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = _string(tool.get("id"))
        media_type = _string(tool.get("type") or tool.get("model_type"))
        provider = _string(tool.get("provider"))
        model = _string(tool.get("model") or tool.get("name") or tool.get("key"))
        if not tool_id and not (media_type and provider and model):
            continue
        lines.append(
            f"- {media_type or 'tool'}: tool_id={tool_id or '(derive)'}, provider={provider}, model={model}"
        )
    return lines


def _model_supports_vision(text_model: Dict[str, Any]) -> bool:
    provider = _string(text_model.get("provider")).lower()
    model = _string(text_model.get("model")).lower()
    vision_models = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "claude-3",
        "claude-sonnet",
        "gemini-pro-vision",
        "gemini-2.5",
        "gemini-3",
        "seed-1-6",
        "llava",
        "bakllava",
    )
    if provider == "ollama":
        return any(name in model for name in ("llava", "bakllava"))
    if provider == "byteplus" and "deepseek" in model:
        return False
    return any(name in model for name in vision_models)


def _filter_image_content(messages: List[Dict[str, Any]], text_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    if _model_supports_vision(text_model):
        return messages
    filtered: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            filtered.append(msg)
            continue
        kept: List[Any] = []
        for item in content:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            item_type = item.get("type")
            if item_type == "image_url":
                continue
            if item_type == "text":
                kept.append(item)
        if kept:
            msg_copy = dict(msg)
            if len(kept) == 1 and isinstance(kept[0], dict) and kept[0].get("type") == "text":
                msg_copy["content"] = kept[0].get("text", "")
            else:
                msg_copy["content"] = kept
            filtered.append(msg_copy)
        elif msg.get("role") != "user":
            filtered.append(msg)
    return filtered


def _fix_chat_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_message_ids = {
        _string(msg.get("tool_call_id"))
        for msg in messages
        if msg.get("role") == "tool" and _string(msg.get("tool_call_id"))
    }
    fixed: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("tool_calls"), list):
            fixed.append(msg)
            continue
        valid_tool_calls = []
        for tool_call in msg.get("tool_calls") or []:
            tool_call_id = _string(tool_call.get("id")) if isinstance(tool_call, dict) else ""
            if tool_call_id and tool_call_id in tool_message_ids:
                valid_tool_calls.append(tool_call)
        if valid_tool_calls:
            msg_copy = dict(msg)
            msg_copy["tool_calls"] = valid_tool_calls
            fixed.append(msg_copy)
        elif msg.get("content"):
            msg_copy = dict(msg)
            msg_copy.pop("tool_calls", None)
            fixed.append(msg_copy)
    return fixed


def _input_image_refs_from_text(text: str) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for tag in re.findall(r"<image\b[^>]*>", text or "", flags=re.IGNORECASE):
        ref: Dict[str, str] = {}
        for key, value in re.findall(r'\b([a-zA-Z0-9_:-]+)="([^"]*)"', tag):
            if value:
                ref[key] = value
        if ref:
            refs.append(ref)
    return refs


def _input_images_from_text(text: str) -> List[Any]:
    images: List[Any] = []
    for ref in _input_image_refs_from_text(text):
        object_name = _string(ref.get("s3_object_name") or ref.get("object_name"))
        if object_name:
            image = {
                "s3_object_name": object_name,
                "file_id": _string(ref.get("file_id")),
                "width": _string(ref.get("width")),
                "height": _string(ref.get("height")),
            }
            for key in ("filename", "mime_type", "role", "reference_note"):
                value = _string(ref.get(key))
                if value:
                    image[key] = value
            images.append(image)
            continue
        file_id = _string(ref.get("file_id"))
        if file_id:
            images.append(file_id)
    return images


def _storybook_page_refs_from_text(text: str) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for match in re.finditer(
        r"<page\b(?P<attrs>[^>]*)>(?P<body>.*?)</page>",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    ):
        ref: Dict[str, Any] = {}
        for key, value in re.findall(r'\b([a-zA-Z0-9_:-]+)="([^"]*)"', match.group("attrs") or ""):
            if not value:
                continue
            if key in ("page_index", "page_number"):
                try:
                    ref[key] = int(value)
                except ValueError:
                    ref[key] = value
            else:
                ref[key] = value
        body = match.group("body") or ""
        narration = _xml_tag_text(body, "narration")
        image_prompt = _xml_tag_text(body, "image_prompt")
        if narration:
            ref["current_narration"] = narration
        if image_prompt:
            ref["current_image_prompt"] = image_prompt
        if ref.get("storybook_id"):
            refs.append(ref)
    return refs


def _storybook_page_update_intent(text: str) -> bool:
    value = (text or "").lower()
    return "<storybook_page_references" in value


def _asset_input_image(asset: Dict[str, Any]) -> Any:
    object_name = _string(asset.get("s3_object_name") or asset.get("object_name") or asset.get("key"))
    if object_name:
        image = {
            "s3_object_name": object_name,
            "file_id": _string(asset.get("file_id") or asset.get("id")),
            "width": _string(asset.get("width")),
            "height": _string(asset.get("height")),
        }
        for key in ("filename", "mime_type", "role", "reference_note"):
            value = _string(asset.get(key))
            if value:
                image[key] = value
        return image
    url = _string(asset.get("url") or asset.get("image_url"))
    return url


def _image_ref_from_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, list):
        for item in reversed(content):
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            raw = item.get("image_url")
            ref = raw if isinstance(raw, dict) else {"url": _string(raw)}
            image = _asset_input_image(ref)
            if image:
                return image
    if message.get("role") == "tool" and "image" in _string(message.get("name")).lower():
        assets = _extract_generated_assets(message.get("content"), "image")
        for asset in reversed(assets):
            image = _asset_input_image(asset)
            if image:
                return image
    return None


def _latest_generated_image_ref(messages: List[Any]) -> Any:
    for message in reversed(messages or []):
        image = _image_ref_from_message(message)
        if image:
            return image
    return None


def _regeneration_intent(value: str) -> bool:
    regeneration_words = (
        "regenerate",
        "re-generate",
        "redo",
        "remake",
        "again",
        "one more",
        "another version",
        "new version",
        "more detail",
        "more details",
        "add detail",
        "enhance",
        "improve",
        "refine",
        "polish",
        "upscale",
        "make it better",
        "重新生成",
        "重新產生",
        "再生成",
        "再產生",
        "重生成",
        "重做",
        "再做",
        "再来",
        "再來",
        "再画",
        "再畫",
        "换一版",
        "換一版",
        "新版本",
        "另一个版本",
        "另一個版本",
        "更多细节",
        "更多細節",
        "加细节",
        "加細節",
        "细节更多",
        "細節更多",
        "增强",
        "增強",
        "优化",
        "優化",
        "改进",
        "改進",
        "改善",
        "修改",
        "调整",
        "調整",
        "精修",
        "精细",
        "精細",
        "高清",
        "清晰",
    )
    return any(word in value for word in regeneration_words)


def _media_intent(text: str, has_image_context: bool = False, has_video_context: bool = False) -> str:
    value = (text or "").lower()
    if not value.strip():
        return ""
    if _media_analysis_intent(value):
        return ""
    creation_words = (
        "generate",
        "create",
        "make",
        "draw",
        "design",
        "render",
        "produce",
        "crea",
        "crear",
        "paint",
        "sketch",
        "illustrate",
        "regenerate",
        "redo",
        "remake",
        "enhance",
        "improve",
        "refine",
        "edit",
        "transform",
        "turn",
        "replace",
        "inpaint",
        "生成",
        "创建",
        "制作",
        "画",
        "绘制",
        "设计",
        "渲染",
        "重新生成",
        "重新產生",
        "再生成",
        "再產生",
        "重做",
        "再做",
        "再来",
        "再來",
        "优化",
        "優化",
        "增强",
        "增強",
        "修改",
    )
    image_words = (
        "image",
        "picture",
        "photo",
        "poster",
        "logo",
        "avatar",
        "illustration",
        "drawing",
        "visual",
        "cover",
        "thumbnail",
        "sticker",
        "icon",
        "图片",
        "图像",
        "照片",
        "海报",
        "头像",
        "插画",
        "图",
    )
    video_words = (
        "video",
        "clip",
        "animation",
        "animate",
        "motion",
        "trailer",
        "seedance",
        "veo",
        "视频",
        "动画",
        "短片",
    )
    audio_words = (
        "audio",
        "speech",
        "voice",
        "voiceover",
        "voice-over",
        "narration",
        "tts",
        "spoken",
        "read aloud",
        "音频",
        "音訊",
        "语音",
        "語音",
        "旁白",
        "朗读",
        "朗讀",
        "粤语",
        "粵語",
        "广东话",
        "廣東話",
    )
    has_creation = any(word in value for word in creation_words)
    has_regeneration = _regeneration_intent(value)
    if not has_creation and not (has_regeneration and (has_image_context or has_video_context)):
        return ""
    if any(word in value for word in audio_words):
        return "audio"
    if any(word in value for word in video_words):
        return "video"
    if any(word in value for word in image_words):
        return "image"
    if has_regeneration:
        if has_video_context:
            return "video"
        if has_image_context:
            return "image"
    if any(word in value for word in ("draw", "render", "paint", "sketch", "illustrate", "画", "绘制")):
        return "image"
    return ""


def _duration_seconds_from_text(text: str) -> int:
    value = text or ""
    patterns = (
        r"\b(\d{1,3})\s*(?:seconds?|secs?|s)\b",
        r"\b(\d{1,3})\s*(?:second|sec)s?\b",
        r"\b(\d{1,3})\s*秒\b",
    )
    for pattern in patterns:
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


def _audio_language_type_from_text(text: str) -> str:
    plain_text = _strip_audio_preferences(text)
    value = plain_text.lower()
    if any(word in value for word in ("粤语", "粵語", "广东话", "廣東話", "cantonese", "yue")):
        return "cantonese"
    if any(word in value for word in ("english", "英文", "英语", "英語")):
        return "english"
    preference = re.search(r"<audio_preferences\b[^>]*\blanguage_type=[\"']([^\"']+)[\"']", text or "", re.I)
    if preference:
        preferred = preference.group(1).strip().lower()
        if preferred in {"mandarin", "cantonese", "english"}:
            return preferred
    if re.search(r"[\u4e00-\u9fff]", plain_text):
        return "mandarin"
    return "english"


def _strip_audio_preferences(text: str) -> str:
    return re.sub(r"\n*\s*<audio_preferences\b[^>]*/>\s*", "\n", text or "", flags=re.I).strip()


def _clean_audio_topic(text: str) -> str:
    value = _strip_audio_preferences(text)
    value = re.sub(
        r"^\s*(/audio|generate\s+(an?\s+)?audio|create\s+(an?\s+)?audio|generate\s+speech|create\s+speech|"
        r"生成一段?音频|生成一段?音訊|生成音频|生成音訊|生成语音|生成語音|生成旁白)\s*[:：,，-]*\s*",
        "",
        value,
        flags=re.I,
    ).strip()
    value = re.sub(r"\b(use|in|with)\s+(mandarin|cantonese|english)\b", "", value, flags=re.I).strip()
    value = re.sub(r"(用|以)?(中文|普通话|普通話|粤语|粵語|广东话|廣東話|英文|英语|英語)(介绍|介紹|朗读|朗讀|讲解|講解)?", "", value).strip()
    return value or _strip_audio_preferences(text).strip()


def _audio_script_from_request(text: str, language_type: str = "") -> str:
    topic = _clean_audio_topic(text)
    language = language_type or _audio_language_type_from_text(text)
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


def _game_intent(text: str) -> bool:
    value = (text or "").lower()
    if not value.strip():
        return False
    return any(
        word in value
        for word in (
            "game",
            "playable",
            "interactive demo",
            "platformer",
            "maze",
            "arcade",
            "boss challenge",
            "storybook game",
            "pokemon-style",
            "pokémon-style",
            "生成游戏",
            "製作遊戲",
            "制作游戏",
            "互动游戏",
            "互動遊戲",
            "闯关",
            "闖關",
            "小游戏",
            "小遊戲",
            "crear juego",
        )
    )


def _storybook_intent(text: str) -> bool:
    value = (text or "").lower()
    if not value.strip():
        return False
    if "<storybook_page_references" in value:
        return False
    return any(
        word in value
        for word in (
            "storybook",
            "story book",
            "flip-book",
            "flip book",
            "page-by-page",
            "children's book",
            "childrens book",
            "picture book",
            "绘本",
            "繪本",
            "故事书",
            "故事書",
            "童书",
            "童書",
            "翻页故事",
            "翻頁故事",
        )
    )


def _agent_max_tokens(config: Dict[str, Any], *, is_game: bool = False) -> Optional[int]:
    configured = _int(config.get("max_tokens"), 0)
    env_name = "ALPHART_AGENT_GAME_MAX_TOKENS" if is_game else "ALPHART_AGENT_MAX_TOKENS"
    env_value = _int(os.getenv(env_name), 0)
    if is_game:
        return max(configured, env_value, 32768)
    if configured > 0:
        return configured
    if env_value > 0:
        return env_value
    return None


def _media_analysis_intent(value: str) -> bool:
    if not re.search(r"<input_(?:images|videos)\b", value) and not any(
        word in value
        for word in (
            "image",
            "picture",
            "photo",
            "video",
            "图片",
            "图像",
            "照片",
            "视频",
        )
    ):
        return False
    analysis_words = (
        "explain",
        "describe",
        "analyze",
        "analyse",
        "summarize",
        "caption",
        "identify",
        "recognize",
        "what is",
        "what's",
        "tell me about",
        "tell me",
        "look at",
        "what does this show",
        "what is shown",
        "解释",
        "说明",
        "说明一下",
        "分析",
        "描述",
        "总结",
        "识别",
        "介绍",
        "讲解",
        "说说",
        "看看",
        "解读",
        "看一下",
        "这是什么",
        "是什么",
        "讲讲",
    )
    return any(word in value for word in analysis_words)


def _selected_media_tools(media_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in _selected_tools():
        tool_type = _string(tool.get("type") or tool.get("model_type")).lower()
        if tool_type == media_type:
            out.append(tool)
    return out


def _generation_tool_called(messages: List[Any], media_type: str) -> bool:
    if media_type == "image":
        expected = "image"
    elif media_type == "video":
        expected = "video"
    else:
        expected = "audio"
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            name = _tool_call_name(tool_call)
            if name in {f"generate_{expected}", f"canvas_generate_{expected}"}:
                return True
            if name.startswith(f"generate_{expected}_by_"):
                return True
    return False


def _generation_tool_completed(messages: List[Any], media_type: str) -> bool:
    if media_type == "image":
        expected = "image"
    elif media_type == "video":
        expected = "video"
    else:
        expected = "audio"
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = _string(msg.get("name")).lower()
        if expected not in name:
            continue
        if _tool_result_success(msg.get("content")):
            return True
    return False


def _generation_tool_attempted(messages: List[Any], media_type: str = "") -> bool:
    expected = media_type.lower().strip()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                name = _tool_call_name(tool_call).lower()
                if not name:
                    continue
                if expected == "image" and ("generate_image" in name or "canvas_generate_image" in name):
                    return True
                if expected == "video" and ("generate_video" in name or "canvas_generate_video" in name):
                    return True
                if expected == "audio" and ("generate_audio" in name or "canvas_generate_audio" in name):
                    return True
                if expected == "game" and ("generate_game" in name or "canvas_generate_game" in name):
                    return True
                if not expected and (
                    "generate_image" in name
                    or "canvas_generate_image" in name
                    or "generate_video" in name
                    or "canvas_generate_video" in name
                    or "generate_audio" in name
                    or "canvas_generate_audio" in name
                    or "generate_game" in name
                    or "canvas_generate_game" in name
                ):
                    return True
        if msg.get("role") == "tool":
            name = _string(msg.get("name")).lower()
            if expected == "image" and "image" in name:
                return True
            if expected == "video" and "video" in name:
                return True
            if expected == "audio" and "audio" in name:
                return True
            if expected == "game" and "game" in name:
                return True
            if not expected and ("image" in name or "video" in name or "audio" in name or "game" in name):
                return True
    return False


def _generation_tool_failed(messages: List[Any], media_type: str = "") -> bool:
    expected = media_type.lower().strip()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = _string(msg.get("name")).lower()
        if expected == "image" and "image" not in name:
            continue
        if expected == "video" and "video" not in name:
            continue
        if expected == "audio" and "audio" not in name:
            continue
        if expected == "game" and "game" not in name:
            continue
        if expected == "" and "image" not in name and "video" not in name and "audio" not in name and "game" not in name:
            continue
        if not _tool_result_success(msg.get("content")):
            return True
    return False


def _generation_tool_effectively_failed(messages: List[Any], media_type: str = "") -> bool:
    return _generation_tool_failed(messages, media_type) and not _generation_tool_completed(messages, media_type)


def _generation_tool_error(messages: List[Any], media_type: str = "") -> str:
    expected = media_type.lower().strip()
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = _string(msg.get("name")).lower()
        if expected and expected not in name:
            continue
        content = msg.get("content")
        try:
            decoded = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict) and decoded.get("success") is False:
            message = _string(decoded.get("error")).strip()
            if message:
                return message[:500]
    return ""


def _game_tool_failed(messages: List[Any]) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = _string(msg.get("name")).lower()
        if "generate_game" not in name and "canvas_generate_game" not in name and "game" not in name:
            continue
        if not _tool_result_success(msg.get("content")):
            return True
    return False


def _storybook_tool_attempted(messages: List[Any]) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                name = _tool_call_name(tool_call).lower()
                if "storybook" in name:
                    return True
        if msg.get("role") == "tool" and "storybook" in _string(msg.get("name")).lower():
            return True
    return False


def _xml_tag_text(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return _string(match.group(1)) if match else ""


def _quantity_from_text(text: str) -> int:
    explicit = _xml_tag_text(text, "image_quantity")
    if explicit.isdigit():
        return max(1, int(explicit))
    match = re.search(r"\b(\d{1,2})\s*(?:images?|pictures?|photos?)\b", text or "", flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _aspect_ratio_from_text(text: str) -> str:
    explicit = _xml_tag_text(text, "aspect_ratio")
    if explicit:
        return explicit
    match = re.search(r"\b(1:1|16:9|9:16|4:3|3:4|2:3|3:2)\b", text or "")
    return match.group(1) if match else ""


def _tool_result_success(result: str) -> bool:
    if _string(result).strip().lower() in {"generate fail", "system busy", SYSTEM_BUSY_MESSAGE.lower()}:
        return False
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return False
    if isinstance(decoded, dict) and decoded.get("success") is False:
        return False
    if isinstance(decoded, dict) and _string(decoded.get("status")).lower() in {"failed", "error", "failure"}:
        return False
    if isinstance(decoded, dict) and isinstance(decoded.get("result"), dict):
        result_status = _string(decoded["result"].get("status")).lower()
        if result_status in {"failed", "error", "failure"}:
            return False
    return True


def _extract_generated_assets(result: Any, media_type: str) -> List[Dict[str, Any]]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return []
    if not isinstance(result, dict):
        return []
    if result.get("success") is False:
        return []
    payload = result.get("result", result)
    candidates: List[Any]
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = [payload]
        for key in ("assets", "images", "videos", "audios", "audio", "data", "outputs", "media"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, dict):
                candidates.append(value)
    else:
        candidates = []

    assets: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        url = _string(item.get("url") or item.get("image_url") or item.get("video_url") or item.get("audio_url"))
        mime_type = _string(item.get("mime_type") or item.get("mimeType"))
        if not url:
            continue
        if media_type == "image" and mime_type and not mime_type.startswith("image/"):
            continue
        if media_type == "video" and mime_type and not mime_type.startswith("video/"):
            continue
        if media_type == "audio" and mime_type and not mime_type.startswith("audio/"):
            continue
        assets.append(item)
    return assets


def _asset_object_name(asset: Dict[str, Any], media_type: str) -> str:
    if media_type == "video":
        return _string(
            asset.get("s3_object_name")
            or asset.get("object_name")
            or asset.get("key")
            or asset.get("video_url_s3_object_name")
        )
    if media_type == "audio":
        return _string(
            asset.get("s3_object_name")
            or asset.get("object_name")
            or asset.get("key")
            or asset.get("audio_url_s3_object_name")
        )
    return _string(
        asset.get("s3_object_name")
        or asset.get("object_name")
        or asset.get("key")
        or asset.get("image_url_s3_object_name")
    )


def _message_has_media_url(messages: List[Any], url: str) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and url in content:
            return True
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url") == url:
                return True
            if _string(item.get("video_url")) == url:
                return True
            if _string(item.get("audio_url")) == url:
                return True
    return False


def _append_visible_generated_media(messages: List[Any], scan_messages: Optional[List[Any]] = None) -> List[Any]:
    out = list(messages or [])
    for msg in scan_messages if scan_messages is not None else messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = _string(msg.get("name")).lower()
        if "image" in name:
            for asset in _extract_generated_assets(msg.get("content"), "image"):
                url = _string(asset.get("url") or asset.get("image_url"))
                if not url or _message_has_media_url(out, url):
                    continue
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": "Generated image."},
                        ],
                    }
                )
        elif "audio" in name:
            for asset in _extract_generated_assets(msg.get("content"), "audio"):
                url = _string(asset.get("url") or asset.get("audio_url"))
                if not url or _message_has_media_url(out, url):
                    continue
                audio_part: Dict[str, Any] = {
                    "type": "generate_audio_result",
                    "audio_url": url,
                }
                object_name = _asset_object_name(asset, "audio")
                if object_name:
                    audio_part["s3_object_name"] = object_name
                mime_type = _string(asset.get("mime_type") or asset.get("mimeType") or "audio/wav")
                if mime_type:
                    audio_part["mime_type"] = mime_type
                duration = asset.get("duration_seconds") or asset.get("duration")
                if duration:
                    audio_part["duration_seconds"] = duration
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            audio_part,
                            {"type": "text", "text": "Generated audio."},
                        ],
                    }
                )
        elif "video" in name:
            for asset in _extract_generated_assets(msg.get("content"), "video"):
                url = _string(asset.get("url") or asset.get("video_url"))
                if not url or _message_has_media_url(out, url):
                    continue
                video_part: Dict[str, Any] = {
                    "type": "video_url",
                    "video_url": url,
                }
                object_name = _asset_object_name(asset, "video")
                if object_name:
                    video_part["s3_object_name"] = object_name
                mime_type = _string(asset.get("mime_type") or asset.get("mimeType") or "video/mp4")
                if mime_type:
                    video_part["mime_type"] = mime_type
                duration = asset.get("duration_seconds") or asset.get("duration") or asset.get("video_duration")
                if duration:
                    video_part["duration_seconds"] = duration
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            video_part,
                            {"type": "text", "text": "Generated video."},
                        ],
                    }
                )
    return out


_MEDIA_URL_RE = re.compile(
    r"https?://[^\s)'\"<>]+(?:\.(?:png|jpe?g|webp|gif|mp4|m4v|mov|f4v|flv|webm|ogg|mp3|opus|aac|flac|wav|pcm))(?:\?[^\s)'\"<>]*)?",
    re.IGNORECASE,
)


def _strip_media_urls_from_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    cleaned = re.sub(
        r"!\[[^\]]*\]\(" + _MEDIA_URL_RE.pattern + r"\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = _MEDIA_URL_RE.sub("", cleaned)
    cleaned = re.sub(r"^\s*(Generated image|Generated video|Generated audio|Image|Video|Audio|图片|图像|视频|影片|音频|音声|生成图片|生成视频|生成音频)\s*[:：]?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_assistant_media_url_text(messages: List[Any]) -> List[Any]:
    out: List[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        msg_copy = dict(msg)
        content = msg_copy.get("content")
        if isinstance(content, str):
            msg_copy["content"] = _strip_media_urls_from_text(content)
        elif isinstance(content, list):
            next_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part = dict(part)
                    part["text"] = _strip_media_urls_from_text(_string(part.get("text")))
                    if not part["text"]:
                        continue
                next_content.append(part)
            msg_copy["content"] = next_content
        out.append(msg_copy)
    return out


def _audio_urls_from_content(content: Any) -> List[str]:
    if not isinstance(content, list):
        return []
    urls: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"audio_url", "audio"}:
            url = _string(item.get("audio_url") or item.get("url"))
            if url:
                urls.append(url)
    return urls


def _transcribed_text_from_result(result: str) -> str:
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return ""
    if not isinstance(decoded, dict):
        return ""
    direct = _string(decoded.get("text"))
    if direct:
        return direct
    nested = decoded.get("result")
    if isinstance(nested, dict):
        return _string(nested.get("text"))
    return ""


def _forced_audio_to_media_pipeline(
    audio_urls: List[str],
    user_message: str,
    response_messages: List[Any],
    scan_messages: Optional[List[Any]] = None,
    input_images: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Multi-step pipeline when user sends audio containing a generation command:
    1. Transcribe audio → text
    2. Refine into professional prompt
    3. Call image/video generation API
    """
    if not audio_urls:
        return []
    current_messages = scan_messages if scan_messages is not None else response_messages
    if _generation_tool_attempted(current_messages, "audio"):
        return []

    audio_url = audio_urls[0]
    transcribe_id = str(uuid.uuid4())
    transcribe_args: Dict[str, Any] = {"audio_url": audio_url, "tool_call_id": transcribe_id}

    print(
        f"[alphart-agent] audio pipeline: transcribing audio_url={audio_url[:80]}",
        flush=True,
    )
    transcribe_result = _handle_alphart_transcribe_audio(transcribe_args)
    transcribed_text = _transcribed_text_from_result(transcribe_result) if _tool_result_success(transcribe_result) else ""

    transcribe_call_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": transcribe_id,
                "type": "function",
                "function": {
                    "name": "canvas_transcribe_audio",
                    "arguments": json.dumps(transcribe_args, ensure_ascii=False),
                },
            }
        ],
    }
    transcribe_result_msg: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": transcribe_id,
        "name": "canvas_transcribe_audio",
        "content": transcribe_result,
    }

    if not transcribed_text:
        return [
            {"role": "assistant", "content": "Plan:\n1. Transcribe audio input to text."},
            transcribe_call_msg,
            transcribe_result_msg,
            {"role": "assistant", "content": "generate fail"},
        ]

    intent = _media_intent(transcribed_text)
    effective_prompt = transcribed_text or user_message

    if not intent:
        return [
            {
                "role": "assistant",
                "content": "Plan:\n1. Transcribe audio input to text.\n2. Respond based on transcribed content.",
            },
            transcribe_call_msg,
            transcribe_result_msg,
            {"role": "assistant", "content": f"Transcribed: {transcribed_text}"},
        ]

    plan_text = (
        f"Plan:\n"
        f"1. Transcribe the audio input to text.\n"
        f"2. Refine the transcribed command into a professional {intent} generation prompt.\n"
        f"3. Call the {intent} generation API with the refined prompt."
    )

    gen_id = str(uuid.uuid4())
    print(
        f"[alphart-agent] audio pipeline: intent={intent} transcribed_len={len(transcribed_text)}",
        flush=True,
    )

    if intent == "image":
        gen_args: Dict[str, Any] = {
            "prompt": effective_prompt,
            "tool_call_id": gen_id,
            "image_quantity": _quantity_from_text(effective_prompt),
        }
        if input_images:
            gen_args["input_images"] = input_images
        aspect_ratio = _aspect_ratio_from_text(effective_prompt)
        if aspect_ratio:
            gen_args["aspect_ratio"] = aspect_ratio
        gen_result = _handle_alphart_generate_image(gen_args)
        gen_tool_name = "canvas_generate_image"
        final_text = "Image generated from your audio command." if _tool_result_success(gen_result) else "generate fail"
    else:
        gen_args = {
            "prompt": effective_prompt,
            "tool_call_id": gen_id,
        }
        aspect_ratio = _aspect_ratio_from_text(effective_prompt)
        if aspect_ratio:
            gen_args["aspect_ratio"] = aspect_ratio
        gen_result = _handle_alphart_generate_video(gen_args)
        gen_tool_name = "canvas_generate_video"
        final_text = "Video generated from your audio command." if _tool_result_success(gen_result) else "generate fail"

    return [
        {"role": "assistant", "content": plan_text},
        transcribe_call_msg,
        transcribe_result_msg,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": gen_id,
                    "type": "function",
                    "function": {
                        "name": gen_tool_name,
                        "arguments": json.dumps(gen_args, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": gen_id,
            "name": gen_tool_name,
            "content": gen_result,
        },
        {"role": "assistant", "content": final_text},
    ]


def _forced_storybook_tool_messages(
	user_message: str,
	input_images: Optional[List[Any]] = None,
	audio_language_type: str = "",
) -> List[Dict[str, Any]]:
    if not _storybook_intent(user_message):
        return []

    call_id = str(uuid.uuid4())
    language = _infer_storybook_language(user_message)
    args: Dict[str, Any] = {
        "topic": user_message,
        "prompt": user_message,
        "tool_call_id": call_id,
        "page_count": 10,
        "language": language,
        "read_aloud_language": _storybook_read_aloud_language(user_message, language, audio_language_type),
        "read_aloud": True,
        "generate_images": True,
        "aspect_ratio": "1:1",
    }
    if input_images:
        args["input_images"] = input_images

    print(
        f"[alphart-agent] forcing storybook creation session_intent=storybook input_images={len(input_images or [])}",
        flush=True,
    )
    result = _handle_alphart_create_storybook(args)
    messages: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Plan:\n1. Create a canvas-native flipbook storybook artifact.\n2. Generate strict 1:1 storybook image pages through the backend.\n3. Return the storybook result in chat and canvas.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "canvas_create_storybook",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "canvas_create_storybook",
            "content": result,
        },
	]
    if not _tool_result_success(result):
        messages.append({"role": "assistant", "content": "generate fail"})
    return messages


def _forced_storybook_page_update_messages(
    user_message: str,
    input_images: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    refs = _storybook_page_refs_from_text(user_message)
    if not refs:
        return []
    target = refs[0]
    call_id = str(uuid.uuid4())
    args: Dict[str, Any] = {
        "storybook_id": target.get("storybook_id"),
        "page_id": target.get("page_id"),
        "page_number": target.get("page_number"),
        "page_index": target.get("page_index"),
        "instructions": user_message,
        "tool_call_id": call_id,
        "aspect_ratio": "1:1",
    }
    args = {key: value for key, value in args.items() if value not in (None, "")}
    refs_for_images = input_images or []
    if not refs_for_images and target.get("image_s3_object_name"):
        refs_for_images = [
            {
                "s3_object_name": target.get("image_s3_object_name"),
                "file_id": target.get("page_id") or f"storybook-page-{target.get('page_number') or target.get('page_index') or 1}",
                "role": "current_storybook_page_reference",
                "reference_note": f"Default reference image for page {target.get('page_number') or ''}".strip(),
            }
        ]
    if refs_for_images:
        args["input_images"] = refs_for_images

    print(
        "[alphart-agent] forcing storybook page update "
        f"storybook_id={args.get('storybook_id')} page_number={args.get('page_number')} input_images={len(refs_for_images)}",
        flush=True,
    )
    result = _handle_alphart_update_storybook_page(args)
    final_text = "Storybook page updated." if _tool_result_success(result) else "generate fail"
    return [
        {
            "role": "assistant",
            "content": (
                "Plan:\n"
                "1. Use the referenced storybook page as the edit target.\n"
                "2. Apply the requested change to that specific page only.\n"
                "3. Regenerate the page illustration and update the canvas flipbook."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "canvas_update_storybook_page",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "canvas_update_storybook_page",
            "content": result,
        },
        {"role": "assistant", "content": final_text},
    ]


def _fallback_storybook_pages(user_message: str) -> List[Dict[str, Any]]:
	topic = _string(user_message)[:180] or "a learning adventure"
	return [
		{
			"page_number": 1,
			"page_type": "cover",
			"layout": "cover",
			"title": "Learning Adventure",
			"narration": f"Open the book and meet a story about {topic}.",
			"image_prompt": f"Square 1:1 warm educational storybook cover about {topic}, child-safe, inviting, no body text, clear main character and learning clue.",
			"metadata": {"story_function": "cover", "visual_evidence": "main character, learning clue, safe setting"},
		},
			{
				"page_number": 2,
				"page_type": "image",
				"layout": "image-page",
				"title": "A Question Appears",
				"narration": "",
				"image_prompt": f"Square 1:1 storybook illustration: a curious protagonist notices a concrete question about {topic}; include visible evidence for the question, consistent character, no text.",
				"metadata": {"story_function": "inciting question", "page_turn_hook": "What will the protagonist discover?"},
			},
			{
				"page_number": 3,
				"page_type": "narration",
				"layout": "text-page",
				"title": "The First Clue",
				"narration": f"The protagonist looks closely and finds the first clue about {topic}.",
				"image_prompt": "",
				"metadata": {"story_function": "read-aloud narration"},
			},
			{
				"page_number": 4,
				"page_type": "image",
				"layout": "image-page",
				"title": "Try It",
				"narration": "",
				"image_prompt": f"Square 1:1 storybook illustration: the protagonist tries a safe hands-on example about {topic}; show the key objects/actions mentioned by the lesson, no text.",
				"metadata": {"story_function": "practice", "visual_evidence": "hands-on example and key objects"},
			},
			{
				"page_number": 5,
				"page_type": "narration",
				"layout": "text-page",
				"title": "What Changed?",
				"narration": f"Something changes, and the protagonist compares what happened before and after.",
				"image_prompt": "",
				"metadata": {"story_function": "cause and effect narration"},
			},
			{
				"page_number": 6,
				"page_type": "image",
				"layout": "image-page",
				"title": "The Idea Clicks",
				"narration": "",
				"image_prompt": f"Square 1:1 storybook illustration: a joyful aha moment where the key idea about {topic} becomes visible through concrete classroom-safe symbols, no text.",
				"metadata": {"story_function": "aha moment", "visual_evidence": "clear concrete symbols"},
			},
			{
				"page_number": 7,
				"page_type": "narration",
				"layout": "text-page",
				"title": "Say It Back",
				"narration": f"Now the protagonist can explain the idea in simple words and invites the reader to try.",
				"image_prompt": "",
				"metadata": {"story_function": "reader reflection"},
			},
			{
				"page_number": 8,
				"page_type": "image",
				"layout": "image-page",
				"title": "Use It",
				"narration": "",
				"image_prompt": f"Square 1:1 storybook illustration: the protagonist uses the lesson about {topic} in a new real-world situation, consistent character, warm ending, no text.",
				"metadata": {"story_function": "transfer", "page_turn_hook": "Can the reader find it too?"},
			},
		{
			"page_number": 9,
			"page_type": "image",
			"layout": "image",
			"title": "Share the Discovery",
			"narration": f"The protagonist shares the discovery and celebrates learning with friends.",
			"image_prompt": f"Square 1:1 closing storybook illustration: protagonist shares the discovery about {topic} with friends in a warm child-safe scene, no text.",
			"metadata": {"story_function": "emotional close"},
		},
		{
			"page_number": 10,
			"page_type": "closing",
			"layout": "back-cover",
			"title": "The End",
			"narration": f"The reader can open the book again and notice {topic} in the world.",
			"image_prompt": f"Square 1:1 back-cover style storybook illustration for {topic}, quiet warm closing image, no body text, consistent style.",
			"metadata": {"story_function": "back cover"},
		},
	]


def _forced_media_tool_messages(
    user_message: str,
    response_messages: List[Any],
    scan_messages: Optional[List[Any]] = None,
    has_image_context: bool = False,
    has_video_context: bool = False,
    input_images: Optional[List[Any]] = None,
    approved_audio_script: str = "",
    forced_intent: str = "",
) -> List[Dict[str, Any]]:
    if _storybook_intent(user_message):
        return []
    if _storybook_page_update_intent(user_message):
        return []
    if _media_analysis_intent(user_message.lower()):
        return []
    intent = forced_intent or _media_intent(user_message, has_image_context=has_image_context, has_video_context=has_video_context)
    if not intent:
        return []
    current_messages = scan_messages if scan_messages is not None else response_messages
    if _generation_tool_completed(current_messages, intent):
        return []

    production_prompt = _canvas_fallback_production_prompt(intent, user_message)

    plan_text = (
        "Plan:\n"
        "1. Use the user's request and referenced media as generation context.\n"
        "2. Create a new generation request instead of reusing an old result.\n"
        "3. Return the newly generated media result."
    )

    if intent == "image":
        quantity = min(_quantity_from_text(user_message), 5)
        aspect_ratio = _aspect_ratio_from_text(user_message)
        print(
            f"[alphart-agent] forcing image generation session_intent={intent} quantity={quantity} "
            f"tool_count={len(_selected_media_tools(intent))}",
            flush=True,
        )
        messages: List[Dict[str, Any]] = [{"role": "assistant", "content": plan_text}]
        success_count = 0
        for task_index in range(1, quantity + 1):
            call_id = str(uuid.uuid4())
            task_prompt = production_prompt
            if quantity > 1:
                task_prompt = f"{production_prompt} (variation {task_index} of {quantity}: vary composition, angle, lighting, or style)"
            args: Dict[str, Any] = {
                "prompt": task_prompt,
                "tool_call_id": call_id,
                "image_quantity": 1,
            }
            if has_image_context and input_images:
                args["input_images"] = input_images
            if aspect_ratio:
                args["aspect_ratio"] = aspect_ratio
            result = _handle_alphart_generate_image(args)
            if _tool_result_success(result):
                success_count += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "canvas_generate_image",
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "canvas_generate_image",
                    "content": result,
                }
            )
        if success_count == quantity:
            final_text = "Image generation has been submitted."
        elif success_count > 0:
            final_text = f"Generated {success_count} of {quantity} requested images."
        else:
            final_text = "generate fail"
        messages.append({"role": "assistant", "content": final_text})
        return messages

    if intent == "audio":
        call_id = str(uuid.uuid4())
        script = _string(approved_audio_script) or _last_assistant_text(response_messages)
        language_type = _audio_language_type_from_text(user_message) or _normalize_audio_language_type(_ctx().get("audio_language_type"))
        if not script or script.strip() == user_message.strip() or script.strip().lower().startswith("plan:"):
            script = _audio_script_from_request(user_message, language_type)
        args = {
            "input": script,
            "tool_call_id": call_id,
            "language_type": language_type,
        }
        print(
            f"[alphart-agent] forcing audio generation session_intent={intent} tool_count={len(_selected_media_tools(intent))}",
            flush=True,
        )
        result = _handle_alphart_generate_audio(args)
        tool_name = "canvas_generate_audio"
        messages = [
            {"role": "assistant", "content": script},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": result,
            },
        ]
        if not _tool_result_success(result):
            messages.append({"role": "assistant", "content": "generate fail"})
        return messages

    call_id = str(uuid.uuid4())
    args = {
        "prompt": production_prompt,
        "tool_call_id": call_id,
    }
    if has_image_context and input_images:
        args["input_images"] = input_images
    duration_seconds = _duration_seconds_from_text(user_message)
    if duration_seconds:
        args["duration_seconds"] = duration_seconds
    aspect_ratio = _aspect_ratio_from_text(user_message)
    if aspect_ratio:
        args["aspect_ratio"] = aspect_ratio
    print(
        f"[alphart-agent] forcing video generation session_intent={intent} tool_count={len(_selected_media_tools(intent))}",
        flush=True,
    )
    result = _handle_alphart_generate_video(args)
    tool_name = "canvas_generate_video"
    final_text = (
        "Video generation has been submitted."
        if _tool_result_success(result)
        else "generate fail"
    )
    return [
        {"role": "assistant", "content": plan_text},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": result,
        },
        {"role": "assistant", "content": final_text},
    ]


def _canvas_fallback_production_prompt(intent: str, user_message: str) -> str:
    """Keep Canvas no-tool-call recovery aligned with its specialised skills."""
    if str(_ctx().get("app_scope") or "").strip().lower() != "canvas":
        return user_message
    if intent == "image":
        return (
            "Create a production-ready Canvas keyframe. Preserve every supplied visual "
            "reference, identity, silhouette, composition, palette, and lighting constraint. "
            "Use one clear focal subject and specify setting, subject, composition, camera, "
            "lighting, material, and finish without adding unrelated elements.\n\n"
            f"User brief: {user_message}"
        )
    if intent == "video":
        duration = int(_ctx().get("duration_seconds") or 0)
        pacing = "Use one clear action and one camera move." if duration <= 5 else (
            "Use setup, action, and a clear resolution." if duration <= 9 else
            "Use compact timed beats with a clear ending and no repetitive motion."
        )
        return (
            "Direct one coherent cinematic Canvas video. Preserve continuity across supplied "
            "frame references; respect first/last-frame constraints; use one deliberate camera "
            f"grammar; {pacing} Keep supplied soundtrack or voice-print references intact.\n\n"
            f"User brief: {user_message}"
        )
    return user_message


def _alphart_agent_prompt(req: AlphartEduChatRequest) -> str:
    tool_lines = _selected_tool_lines(req.tool_list)
    selected_tools = "\n".join(tool_lines) if tool_lines else "- No configured canvas media tools are available. Do not invent provider/model names; return a concise configuration error for media generation requests."
    return f"""
{req.system_prompt.strip()}

CANVAS AGENT ROLE:
You are replacing the old planner + image_video_creator LangGraph swarm.
You must preserve both behaviors:
1. Planner behavior: understand the user request, write an execution plan when the task is complex, and route media tasks to generation immediately.
2. Image/video creator behavior: write professional image/video prompts, call the selected generation tools, and explain tool results.

PLANNER RULES:
- Answer and write plans in the same language as the user's prompt.
- For normal conversation, answer directly without calling tools.
- When writing math, physics, chemistry, or engineering formulas, output valid Markdown math. Use inline math as `$...$` and display math as `$$...$$` on separate lines. Do not output raw LaTeX formulas without delimiters, do not double-escape backslashes, and do not emit literal `\n` escape sequences inside prose. Put each standalone formula, such as `y=\pm \frac{{b}}{{a}}x` or `c^2=a^2+b^2`, in its own display math block.
- If the user asks to explain, describe, analyze, summarize, caption, identify, or understand an attached image/video, answer with the text/chat model. Do not call image/video generation tools.
	- For obvious image/video/audio generation or editing tasks, a generation tool call is mandatory.
	- For simple media requests, call canvas_generate_image/canvas_generate_video/canvas_generate_audio directly. Do not stop after a plan.
	- Use the selected tool metadata for provider/model. Do not invent provider/model names and do not rely on backend-selected defaults. If no selected image/video/audio tool is listed for the requested capability, return a concise configuration error.
- For complex media requests, you may call write_plan first, but you must continue to the generation tool after the plan result.
- For Canvas requests where the user asks you to create/manage canvas nodes, use canvas_create_node/canvas_update_node/canvas_connect_nodes. For image generation on Canvas, create an image node first with the enriched professional prompt, then call canvas_generate_image with that node's canvas_item_id so the backend updates the same node with the generated asset.
- If you create a planning/prompt node and a final media node, connect them with canvas_connect_nodes after both node ids are known.
- Do not ask for approval before media generation unless the backend returns a confirmation request.
- Do not call multiple tools in the same assistant turn. Always wait for one tool result before making another tool call.
- If a tool call fails, explain the error to the user and do not retry automatically.
- Pay attention to requested quantity. If the user asks for 20 images, keep exactly 20 in the plan and generation batches. If no quantity is specified, assume 1.
- If the user requests N images (N > 1) for the same theme, you MUST treat this as N separate generation tasks, e.g. "Image 1: <prompt for image 1>", "Image 2: <prompt for image 2>", ... "Image N: <prompt for image N>". Each task should have its own distinct, professionally written prompt (vary composition, angle, lighting, or style so the N images are not identical). After writing the task list, call the image generation tool ONCE PER TASK, one tool call per turn, continuing across turns until all N tool calls have been made and all N images are returned. Do not stop after the first image when more are requested.

SELECTED CANVAS TOOLS:
{selected_tools}

IMAGE CREATION RULES:
- For image generation, call generate_image or canvas_generate_image. Do not wait for approval.
- If you write a Design Strategy Doc, keep it concise and then call the image tool in the same task flow.
- Use a detailed, professional prompt based on the strategy.
- Respect <aspect_ratio>, <image_quantity>, and other XML tags in the user message.
- If the user requests more than 5 images, generate in batches of at most 5. Complete each batch before starting the next batch.
- When the user message contains <input_images> XML, extract s3_object_name values and pass them as input_images. Use file_id only as a fallback.
- If the user asks to regenerate, redo, edit, transform, add more details, enhance, improve, or create a new image using a reference image, previous generated images are only references/history. You must call a fresh image generation tool and must not present an old image URL as the new result.
- Treat equivalent Simplified/Traditional Chinese commands as regeneration/editing intent, including 重新生成, 重新產生, 再生成, 再產生, 重做, 再做, 再来, 再來, 换一版, 換一版, 更多细节, 更多細節, 优化, 優化, 增强, 增強, 修改, 调整, 調整.
- If the user asks to regenerate or add details but does not attach a new image, use the most recent generated image in the session as the reference input image.
- If more than one input image is present, prefer a selected image tool that supports multiple input_images.
- If the request includes facial expression, mood, emotion, age, gender, region, or cultural constraints, add precise expression-control keywords to the prompt and avoid unsafe or culturally forbidden expression details.

	VIDEO CREATION RULES:
- Use video generation tools for video tasks.
- You may generate needed storyboard/keyframe images first, then call video generation using those images, or directly generate video from text if that better fits the request.
- If input images are provided, pass s3_object_name values as input_images. Use file_id only as a fallback.
- Respect duration, resolution, aspect ratio, camera movement, and shot references from XML tags.
- Do not claim media was generated until the tool returns a backend result.
	- If the legacy prompt mentions generate_image, call generate_image or canvas_generate_image. If it mentions generate_video, call generate_video or canvas_generate_video.

		AUDIO CREATION RULES:
		- Use canvas_generate_audio or generate_audio for spoken-audio tasks, including "generate an audio", "create a voiceover", "read aloud", "生成一段音频", "生成一段音訊", "生成语音", "生成語音", "用粤语/粵語/广东话/廣東話介绍", and equivalent requests.
		- Audio generation must produce two user-visible outputs: first a normal assistant text message containing the educational narration/script, then the generated audio result. Do not replace the script with a plan.
		- The audio tool input must be the same ready-to-speak script text from the assistant message, not the raw command.
		- Match the requested spoken language: language_type="cantonese" for 粤语/粵語/广东话/廣東話/Cantonese, language_type="mandarin" for 中文/普通话/普通話/Mandarin, and language_type="english" for English.
		- Do not ask the user to choose an audio model. Use the selected audio tool metadata from SELECTED CANVAS TOOLS, including provider and model.

STORYBOOK CREATION RULES:
- Use canvas_create_storybook or create_storybook for requests like "make a storybook", "create a flip-book lesson", "storybook about ...", "page-by-page children's book", and equivalent Chinese/Traditional Chinese requests such as 绘本, 繪本, 故事书, 故事書, 童书, 童書, 翻页故事, 翻頁故事.
- A storybook is a Gemini-style Edu-native canvas/chat artifact: a complete illustrated flipbook with short page text and read-aloud narration, not a draft, planning document, or mirrored alphart-book API.
- Never create storybooks as HTML files, local files, documents, code projects, or /tmp outputs. Do not call Write, write_file, patch, terminal, process, Bash, or any file/coding tool for storybook requests.
- The only valid tool path for creating a storybook is canvas_create_storybook/create_storybook. The only valid tool path for revising one page is canvas_update_storybook_page/update_storybook_page.
- For storybook requests, first load the English storybook-generator skill with skill_view("storybook-generator"), then call the storybook tool and treat it as the final storybook artifact only if the required image pages are generated. Default to 10 pages, read_aloud=true, generate_images=true, and aspect_ratio="1:1" unless the user explicitly asks for another length.
- For storybook page planning, read the skill references story-structure.md, character-continuity.md, prompt-workflow.md, and qa-checklist.md with skill_view("storybook-generator", "references/<file>"). Read layout-and-pinyin.md only for text layout/export requests, and read commercial-publishing-workflow.md only for KDP/commercial publishing requests.
- After loading the skill guidance, apply its compact workflow: 1) story architecture, 2) character/style bible, 3) page-by-page causality and page-turn hooks, 4) visual evidence contract, 5) page image prompts, 6) child-safety and factual QA. Do not stop after skill_view and do not output only a markdown plan.
- The storybook-generator skill is used for planning/prompts/QA only in Alphart Edu. Do not call generic image tools, local file tools, Write, Bash, or HTML/export workflows for storybook image pages. The required image pages are generated by the backend when canvas_create_storybook/create_storybook is called with generate_images=true.
- Pass an explicit native pages array to canvas_create_storybook/create_storybook. Do not pass pages as a JSON string, markdown block, or prose. Do not rely on backend filler pages when the user gave a real story premise. Keep page objects compact: page_number, page_type, layout, title, narration, and image_prompt for visual pages. Omit metadata unless essential.
- Alphart Edu storybooks use square 1:1 physical pages. Override the generic skill's "one image per page" rule with this app-specific rhythm: cover is an image; back cover is an image; inner left/odd story pages are image pages with image_prompt; inner right/even story pages are narration/text pages with no image_prompt. If a shorter page count is requested, keep the same rhythm compactly.
- Every storybook must have a causal chain: previous page state -> current trigger/action/discovery -> next page hook. Avoid disconnected pretty pictures.
- Every visible noun/action mentioned in narration must have matching visual evidence in the image_prompt, and every image_prompt must support the narration. Do not add unsupported facts or visuals.
- Build character continuity into every page prompt: fixed protagonist appearance, clothing/accessory anchors, expression baseline, scene/world anchors, and reference image roles when present.
- If reference images are provided with @file or <input_images>, use them as protagonist/background/object references according to the user's label, preserve s3_object_name, and do not convert them to base64.
- Do not ask the image model to render normal body text inside illustrations. Use no-text illustrations except short signs/labels essential to the scene. Cover title text may be requested only when short and must be checked.
- Keep page narration short, age-appropriate, safe, and educational. Maintain requested language, bilingual/trilingual intent, reading level, and factual precision.
- Storybook read-aloud defaults are strict: Chinese storybooks use Mandarin narration/audio by default, and English storybooks use English narration/audio by default. Use Cantonese/粤语/粵語/广东话/廣東話 narration only when the user explicitly asks to read/narrate the storybook in Cantonese, such as “用粤语朗读” or “用广东话朗读”. Merely mentioning Cantonese without read-aloud intent is not enough.
- Storybook physical pages and generated illustrations must be strict square 1:1 pages for printer compatibility. Do not use 4:3, 3:4, or widescreen storybook pages.
- If the user selects or names a template, pass compact template fields such as template_slug, template_name, category, age_range, page_count, style, and read_aloud to canvas_create_storybook/create_storybook instead of hiding them inside prose.
- If the user attaches images while asking for a storybook, treat those images as protagonist/character references. Pass them to canvas_create_storybook/create_storybook as input_images and preserve s3_object_name values. Do not call image generation merely because reference images are attached.
- Use protagonist references to keep the main character visually consistent across pages. Include concise protagonist notes when useful.
- If the user references a storybook page with @page 1, @<page 1>, page 1, 第1页, 第1頁, or similar and asks to fix/replace/revise/change the page image, narration, protagonist, or layout, use canvas_update_storybook_page/update_storybook_page instead of creating a new storybook.
- If the user message contains <storybook_page_references>, treat it as an existing storybook page edit target. Use storybook_id, page_id/page_number/page_index, current narration, current image_prompt, and image_s3_object_name from that block. If no new <input_images> are provided, use the referenced page image_s3_object_name as the default visual reference.
- When updating a storybook page, read storybook_id and page records from the previous storybook tool result in chat history. Pass page_number for human references like @page 1. If the user references @reference image or attached media, pass that media as input_images and preserve s3_object_name.
- Do not answer “done” for storybook page edits unless the update_storybook_page tool succeeds.
- Keep content age-appropriate and educational. Preserve factual accuracy, requested language, age range, reading level, page count, narration tone, visual style, and the user's requested protagonists.
- Do not call canvas_generate_game just because the word "story" appears. Use game tools only when the user asks for playable interaction.

GAME CREATION RULES:
- Use canvas_generate_game or generate_game for requests like "make a game", "interactive demo", "quiz game", "platformer", "storybook game", "GBA/Pokemon-style educational battle", "create a playable teaching activity", and equivalent Chinese/Traditional Chinese/Spanish requests such as 生成游戏, 製作遊戲, 互动游戏, 互動遊戲, 闯关, 闖關, 小游戏, 小遊戲, crear juego.
- Only call canvas_generate_game or generate_game for game artifacts. Do not call file-writing or coding tools such as Write, Edit, MultiEdit, Bash, write_file, patch, terminal, or process; they are unavailable in this service and will fail.
- For game requests, first load the English gaming skill with skill_view("gaming"), then follow its studio workflow before calling canvas_generate_game/generate_game. If skill_view reports that gaming is unavailable, continue using the GAME CREATION RULES below as the authoritative fallback instead of stopping or asking the user. Do not stop after skill_view and do not output only a markdown plan.
- For game planning, read the gaming references when useful with skill_view("gaming", "game-studio.md"), skill_view("gaming", "studio-design.md"), skill_view("gaming", "studio-dev.md"), and skill_view("gaming", "studio-qa.md"). Apply them as internal studio reasoning, not as external shell/file commands.
- Game generation must follow a mini version of the game-studio flow: 1) Studio Design: define learning goal, player fantasy, game pattern, content_facts, controls, and win/fail state; 2) Studio Planning: create acceptance criteria for bounded layout, controls, loop, state changes, validation/collision, and completion; 3) Studio Development: create complete playable HTML/CSS/JS, either self-contained in html or as an artifact_dir/artifact_path containing index.html and assets; 4) Studio QA: mentally playtest before calling the tool; 5) call canvas_generate_game/generate_game with prompt plus html or artifact_dir/artifact_path, adding only concise game_plan/layout_requirements/review_checklist fields if useful.
- Default to a simple pixel-art game, not a plain web form. Use blocky sprites, tile/grid playfields, crisp edges, limited high-contrast palettes, HUD panels, and 8-bit inspired controls.
- Prefer real playable patterns: pixel platformer, top-down maze/exploration, arcade matcher, drag-and-drop sorter, physics launcher, simulation sandbox, boss challenge, or story quest. Use a plain quiz/card/form only if the user explicitly asks for a quiz.
- This is an education platform: preserve content precision over visual novelty. Formulas, units, definitions, names, dates, symbols, vocabulary, causal relationships, and domain constraints must be correct.
- This is also a student-safe education platform: do not include vulgar/profane language, sexual content, graphic violence, gore, hate/harassment, self-harm, drug abuse, gambling, extremist content, humiliation, or age-inappropriate jokes. If the game has enemies, attacks, hazards, or penalties, make them non-graphic classroom-safe metaphors such as shields, puzzles, energy, obstacles, misconception blockers, or abstract pixel effects.
- If the source content is ambiguous, avoid inventing details. Use only stable facts from the user request and generally accepted knowledge; mark uncertainty in plain language when necessary.
- The plan must include learning goal, target audience, precise knowledge points, a content_facts ledger, answer option groups with correct choices, likely misconceptions, selected game pattern, core loop, player controls, hazards/collectibles/targets, rules, screen states, pixel-art style, layout grid, safe area, asset list, and win/fail/completion states.
- Every educational label, collectible, hazard, correct answer, wrong answer, dialog, and feedback message must trace to the content_facts ledger, the user request, or stable common knowledge. Do not invent numbers, formulas, dates, names, properties, or causal claims for gameplay.
- Every question or answer-option set must include at least one correct choice. Single-answer questions must have exactly one correct choice. Multi-select questions must clearly say multi-select and have one or more correct choices. Never show an impossible question with no right answer.
- Map correct knowledge into mechanics carefully: collectibles should represent correct facts, hazards should represent specific misconceptions, and feedback must explain why an answer/action is correct or wrong.
- The game must have an actual loop: the player moves/chooses/collects/avoids/solves, receives immediate feedback, and advances toward score, timer, level, or completion.
- Prove the game is playable in the HTML/JS: start/restart buttons must have real event handlers; keyboard, mouse, touch, or button input must mutate state and update the visible UI; score/progress/timer/level/player position/selected answer must change through play; collision, answer checking, or target validation must be implemented; and win/fail/completion must be reachable by playing.
- Do not ship fake gameplay: no static-only cards, placeholder TODO logic, decorative sprites that never affect state, stub handlers, fake progress, or instructions that describe controls/mechanics missing from the code.
- The layout requirements must explicitly require all content, labels, controls, sprites, dialogs, score panels, and buttons to stay inside the visible frame and their parent borders.
- Use an exact fixed 1920x1080 logical game window/stage. The game HTML itself must scale that exact stage to fit the actual browser/iframe viewport, so opening the public game URL in a new tab shows the complete game without scrolling.
- Use a single visible stage/root container with CSS/JS like: html/body width:100%, height:100%, margin:0, overflow:hidden; * box-sizing:border-box; body display:grid/place-items:center; stage width:1920px, height:1080px, overflow:hidden, transform-origin:top left; compute scale = min(innerWidth/1920, innerHeight/1080) and apply transform:scale(scale).
- Keep every sprite, row/column, HUD panel, dialog, modal/window, widget, tooltip, button, and label inside its visible game frame. Include padding and borders in that budget.
- Avoid negative offsets, position:fixed, fixed-position overlays, viewport-sized panels, oversized absolute dialogs, or transforms that push UI outside the game frame.
- Review the result for content correctness, readable instructions, no clipped text, no overlapping elements, no overflow outside cards/frame/borders, usable controls, start/restart flow, win/fail state, and fit at 1920x1080, 1366x768, 1024x768, and phone-sized iframe viewports.
- Reject boring outputs: if it is only a static explanation, a button list, or a form-like quiz, redesign it as a playable pixel game before calling the tool.
- Reject inaccurate outputs: if any educational statement is wrong, vague enough to mislead, or conflicts with the user's content, fix it before calling the tool.
- If the game result has any layout or content issue, do not present it as finished. Revise the prompt with concrete fixes and call the game tool again.
- Do not use image/video generation tools for playable game requests unless the game plan explicitly needs a static asset first.
- The game tool uploads the HTML from the agent. Never call the game tool with only a prompt. The html argument must be a full document beginning with <!DOCTYPE html> and ending with </html>.
- The HTML body should include visible first-paint game DOM content directly in the markup when possible: a root game container, playfield or canvas/SVG, HUD/score/progress, instructions, and start/restart or control elements. JavaScript may enhance or render the playfield.
- Before calling canvas_generate_game/generate_game, verify the standalone URL behavior mentally: no document scrollbars, no clipped game area, no button/dialog outside the frame, no fixed-position element, and the whole 1920x1080 stage scales down as one unit.
- Required QA acceptance criteria before calling the tool: the game is bounded; controls are wired; keyboard/mouse/touch or button input mutates state; an update/render loop or equivalent event-driven game loop exists; score/progress/level/timer/lives/health or player state visibly changes; win/fail/completion can be reached by playing.
- Keep generated HTML compact enough for a tool argument, but complete. Avoid comments, large inline data, verbose prose, unused CSS, and repeated plan/checklist JSON. Do not omit <body>, script, controls, or closing tags.
- If canvas_generate_game returns a validation error such as missing visible DOM, overflow, clipped text, or layout issue, regenerate corrected HTML once using the error text as a hard requirement, then call canvas_generate_game again. This is the one allowed automatic retry exception for game validation.

UPLOADED AUDIO INPUT RULES:
- These rules apply only when the user attaches or references an existing audio file as input.
- When the user message contains an audio_url content part, call canvas_transcribe_audio immediately to get the text.
- After transcription, read the transcribed text, detect the user's intent, and act on it exactly as if the user had typed that text.
- If the transcribed text is an image, video, audio, storybook, or game generation command, follow the corresponding generation rules above.
- If transcription fails, report the failure clearly and stop.

ERROR HANDLING:
- Read backend tool errors carefully.
- Never retry the same failing tool call automatically.
- Exception: for canvas_generate_game validation errors, one corrected retry is allowed as described in GAME CREATION RULES.
- Never call the same tool with the same parameters again without user confirmation.
- Explain the specific failure and suggest a safer alternative prompt or model.
""".strip()


def _system_prompt(req: AlphartEduChatRequest) -> str:
    if _request_app_scope(req) == "canvas":
        return _canvas_agent_prompt(req)
    return _alphart_agent_prompt(req)


def _canvas_agent_prompt(req: AlphartEduChatRequest) -> str:
    """Keep Canvas turns focused on the selected graph item and its references.

    The Edu prompt contains storybook and game workflows which are useful in that
    product but make Canvas media actions unnecessarily indirect.
    """
    tool_lines = _selected_tool_lines(req.tool_list)
    selected_tools = "\n".join(tool_lines) if tool_lines else (
        "- No configured Canvas tools are available. Return a concise configuration error "
        "instead of inventing a provider or model."
    )
    graph_skill_guidance = _canvas_graph_skill_guidance(req)
    workflow_guidance = _canvas_workflow_guidance(req)
    shot_breakdown_instruction = ""
    if _canvas_shot_breakdown_intent(req):
        shot_breakdown_instruction = """
SHOT BREAKDOWN EXECUTION:
- This is analysis only, never a media-generation request.
- Call video_analyze exactly once using the completed Canvas video URL supplied in the system context.
- After the tool succeeds, return exactly the <canvas-shot-breakdown> JSON block required by the preloaded workflow. Do not call skill_view or any Canvas generation tool.
""".strip()
    return f"""
{req.system_prompt.strip()}

CANVAS AGENT ROLE:
You operate a visual Canvas graph. Be decisive, concise, and execute the requested
operation rather than returning a plan for ordinary node work. The current node and
its connected references supplied by the backend are authoritative.

NODE OWNERSHIP RULES:
- When canvas_item_id is present, operate ONLY on that existing node. Never create,
  replace, or connect another node unless the user explicitly asks to do so.
- When canvas_item_id is absent and the user asks to create content, use
  the Canvas Graph Skill: understand the request, create a Prompt text node with the
  enriched prompt, create the requested output node, connect the graph, and generate
  only into the output node. Do not stop after creating nodes.
- When canvas_item_id is absent but a selected text node is supplied in graph context
  and the user asks for media, use it as an input and create a downstream media graph.
- Treat reference_item_ids and the supplied connected node context as the complete
  set of references. A Prompt node is an intentional persisted design artifact for a
  new media graph; do not create extra temporary caption or soundtrack nodes.
- Use canvas_update_node only to change the requested existing node. Never expose
  internal ids, organisation ids, credentials, or storage keys in the user response.

MEDIA RULES:
- The selected Canvas workflow is preloaded below. Apply it before dispatching
  media; do not skip it, expose it as a planning document, or substitute an
  external CLI, API key, local file, or non-Canvas storage path.
- Do not call skills_list or skill_view for Canvas image/video workflows: their
  guidance is already embedded below and may not appear in the user's optional
  skill inventory.
- For an image, video, or audio generation request, call the matching Canvas tool
  immediately. Use the selected provider/model metadata and do not invent values.
- Preserve the requested duration, ratio, quality, and model. For video, pass the
  exact requested duration (5-15 seconds) when supplied.
- Canvas generates the approved voiceover and burns its SRT separately. Never put
  caption or dialogue text into a video-provider prompt. With no soundtrack/BGM
  reference, let the video provider generate ambient audio; with a soundtrack/BGM
  reference, pass that reference and disable provider-generated audio.
- For audio generation, first produce a ready-to-speak script in the requested
  language, then call canvas_generate_audio with that exact script. The script must
  fit the requested duration and must not be a generic status message.
- If script_only is set, produce only the requested script or text refinement; do
  not call a media or node tool.
- If a tool fails, report the specific failure without overwriting existing node
  content and do not automatically retry the same request.

CONVERSATION RULES:
- Answer normal questions directly without tools.
- Do not use Edu-only workflows such as storybooks, games, course artefacts, or
  file-writing tools in Canvas.
- Do not ask for approval before a straightforward generation request.
- Do not claim media exists until the tool returns a successful result.
- Keep the final response short: confirm what changed or state the actionable error.

SELECTED CANVAS TOOLS:
{selected_tools}

CANVAS GRAPH SKILL:
{graph_skill_guidance or 'No Canvas graph skill is available; follow the strict node ownership and API rules above.'}

PRELOADED CANVAS WORKFLOW:
{workflow_guidance or 'No specialised workflow applies to this node type.'}

{shot_breakdown_instruction}
""".strip()


_canvas_workflow_cache: Dict[str, str] = {}


def _canvas_shot_breakdown_intent(req: AlphartEduChatRequest) -> bool:
    if _canvas_workflow_item_type(req) != "video":
        return False
    text = _canvas_request_text(req).lower()
    return any(phrase in text for phrase in (
        "video-shot-breakdown", "shot breakdown", "shot-by-shot", "shot by shot",
        "拉片", "分镜分析", "镜头分析", "镜头拆解",
    ))


def _canvas_video_shotcraft_intent(req: AlphartEduChatRequest) -> bool:
    if _request_app_scope(req) != "canvas" or _canvas_workflow_item_type(req) != "video":
        return False
    text = _canvas_request_text(req).lower()
    return any(phrase in text for phrase in (
        "video-shotcraft", "shotcraft", "shot recipe", "shot card", "镜头配方", "镜头卡",
    ))


def _canvas_workflow_guidance(req: AlphartEduChatRequest) -> str:
    skill_by_item_type = {
        "video": ("canvas-seedance2-video-director", "seedance2-video-director"),
    }
    if _canvas_shot_breakdown_intent(req):
        skill_by_item_type["video"] = ("canvas-video-shot-breakdown", "video-shot-breakdown")
    elif _canvas_video_shotcraft_intent(req):
        skill_by_item_type["video"] = ("canvas-video-shotcraft", "video-shotcraft")
    skill = skill_by_item_type.get(_canvas_workflow_item_type(req))
    if not skill or req.script_only:
        return ""
    skill_name, skill_directory = skill
    if skill_name in _canvas_workflow_cache:
        return _canvas_workflow_cache[skill_name]
    try:
        from tools.skills_tool import skill_view

        payload = json.loads(skill_view(skill_name, preprocess=True))
        content = _string(payload.get("content")) if isinstance(payload, dict) and payload.get("success") else ""
        if content:
            _canvas_workflow_cache[skill_name] = content
            return content
    except Exception as exc:
        logger.warning("failed to preload Canvas workflow skill %s: %s", skill_name, exc)
    # Canvas workflows ship with this service and are mandatory product
    # guidance, not optional user-installed skills. The user skill inventory
    # may deliberately omit them, so load the bundled source directly.
    bundled_skill = Path(__file__).resolve().parent / "skills" / "canvas" / skill_directory / "SKILL.md"
    try:
        content = bundled_skill.read_text(encoding="utf-8").strip()
        if content:
            _canvas_workflow_cache[skill_name] = content
            return content
    except OSError as exc:
        logger.warning("failed to read bundled Canvas workflow %s: %s", skill_name, exc)
    return ""


def _canvas_request_text(req: AlphartEduChatRequest) -> str:
    return "\n".join(_message_text(message) for message in req.messages if isinstance(message, dict)).strip()


def _canvas_workflow_item_type(req: AlphartEduChatRequest) -> str:
    """Resolve Canvas workflow capacity without changing Edu intent routing."""
    if _request_app_scope(req) != "canvas":
        return ""
    selected_type = _string(
        getattr(req, "canvas_item_type", "") or getattr(req, "selected_canvas_item_type", "")
    ).strip().lower()
    text = _canvas_request_text(req)
    if selected_type in {"text", "note"}:
        requested = _media_intent(text, has_image_context=bool(req.input_images))
        if requested:
            return requested
        return selected_type
    if selected_type in {"image", "video", "audio"}:
        return selected_type
    lowered = text.lower()
    if any(phrase in lowered for phrase in (
        "video-shot-breakdown", "shot breakdown", "shot-by-shot", "shot by shot",
        "拉片", "分镜分析", "镜头分析", "镜头拆解",
    )):
        return "video"
    if any(phrase in lowered for phrase in (
        "video-shotcraft", "shotcraft", "shot recipe", "shot card", "镜头配方", "镜头卡",
    )):
        return "video"
    return _media_intent(text, has_image_context=bool(req.input_images), has_video_context=False)


def _canvas_graph_skill_guidance(req: AlphartEduChatRequest) -> str:
    if req.script_only:
        return ""
    cache_key = "canvas-graph"
    if cache_key in _canvas_workflow_cache:
        return _canvas_workflow_cache[cache_key]
    bundled_skill = Path(__file__).resolve().parent / "skills" / "canvas" / "graph" / "SKILL.md"
    try:
        content = bundled_skill.read_text(encoding="utf-8").strip()
        if content:
            _canvas_workflow_cache[cache_key] = content
            return content
    except OSError as exc:
        logger.warning("failed to read bundled Canvas graph skill: %s", exc)
    return ""


def _is_internal_tool_view_name(name: str) -> bool:
    return _string(name).strip().lower() in {"skill_view", "tool_view", "tool_views"}


def _public_messages(messages: List[Any]) -> List[Any]:
    out: List[Any] = []
    hidden_tool_call_ids = {
        _string(tool_call.get("id"))
        for msg in messages or []
        if isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("tool_calls"), list)
        for tool_call in msg.get("tool_calls") or []
        if isinstance(tool_call, dict) and _is_internal_tool_view_name(_tool_call_name(tool_call))
    }
    hidden_tool_call_ids.discard("")
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        cleaned = {k: v for k, v in msg.items() if not str(k).startswith("_")}
        if cleaned.get("role") == "tool":
            if _string(cleaned.get("tool_call_id")) in hidden_tool_call_ids:
                continue
            if _is_internal_tool_view_name(cleaned.get("name") or cleaned.get("tool_name")):
                continue
        if cleaned.get("role") == "assistant" and isinstance(cleaned.get("tool_calls"), list):
            tool_calls = [
                tool_call
                for tool_call in cleaned.get("tool_calls") or []
                if not _is_internal_tool_view_name(_tool_call_name(tool_call))
            ]
            if tool_calls:
                cleaned["tool_calls"] = tool_calls
            else:
                cleaned.pop("tool_calls", None)
                if not _string(cleaned.get("content")).strip():
                    continue
        if "content" in cleaned:
            cleaned["content"] = _message_text(cleaned)
        out.append(cleaned)
    return _prefer_storybook_artifact_messages(out)


def _prefer_storybook_artifact_messages(messages: List[Any]) -> List[Any]:
    has_storybook_artifact = any(_is_successful_storybook_tool_message(msg) for msg in messages or [])
    if not has_storybook_artifact:
        return messages
    out: List[Any] = []
    for msg in messages or []:
        if _is_failed_storybook_tool_message(msg):
            continue
        out.append(msg)
    return out


def _is_storybook_tool_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "tool":
        return False
    name = _string(message.get("name") or message.get("tool_name")).lower()
    return "storybook" in name


def _storybook_tool_payload(message: Any) -> Dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _is_successful_storybook_tool_message(message: Any) -> bool:
    if not _is_storybook_tool_message(message):
        return False
    payload = _storybook_tool_payload(message)
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return False
    return _string(result.get("type")).lower() in {"storybook", "storybook_page_update"}


def _is_failed_storybook_tool_message(message: Any) -> bool:
    if not _is_storybook_tool_message(message):
        return False
    payload = _storybook_tool_payload(message)
    if not payload:
        return False
    if payload.get("success") is False:
        return True
    return _string(payload.get("error")) != "" and not isinstance(payload.get("result"), dict)


def _storybook_result_record(value: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(decoded, dict):
            payload = decoded
    if not payload:
        return {}
    result = payload.get("result")
    if isinstance(result, dict):
        canvas_element = result.get("canvas_element")
        if isinstance(canvas_element, dict):
            custom_data = canvas_element.get("customData")
            if isinstance(custom_data, dict):
                return custom_data
        return result
    return payload


def _storybook_completion_text(messages: List[Any]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        candidates: List[Any] = []
        if msg.get("role") == "tool" and "storybook" in _string(msg.get("name") or msg.get("tool_name")).lower():
            candidates.append(msg.get("content"))
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tool_call in msg.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                if "storybook" in _tool_call_name(tool_call).lower():
                    candidates.append(tool_call.get("result"))
        for candidate in candidates:
            record = _storybook_result_record(candidate)
            if _string(record.get("type")).lower() not in {"storybook", "storybook_page_update"}:
                continue
            storybook_id = _string(record.get("storybook_id") or record.get("id"))
            if not storybook_id:
                continue
            title = _string(record.get("title") or record.get("topic") or "storybook")
            status = _string(record.get("status") or "completed")
            pages = record.get("pages") if isinstance(record.get("pages"), list) else []
            page_count = int(record.get("page_count") or len(pages) or 0)
            image_pages = sum(1 for page in pages if isinstance(page, dict) and _string(page.get("image_s3_object_name") or page.get("image") or page.get("image_url")))
            text_pages = max(0, page_count - image_pages) if page_count else 0
            if _string(record.get("type")).lower() == "storybook_page_update":
                return f"Storybook page update completed.\n\n- Storybook ID: `{storybook_id}`\n- Status: `{status}`"
            lines = [f"Storybook created: {title}"]
            if page_count:
                lines.append(f"- Pages: {page_count}")
            if image_pages:
                lines.append(f"- Illustration pages: {image_pages}")
            if text_pages:
                lines.append(f"- Narration/text pages: {text_pages}")
            lines.append(f"- Storybook ID: `{storybook_id}`")
            return "\n".join(lines)
    return ""


def _last_assistant_text_after_last_tool(messages: List[Any]) -> str:
    start = 0
    for idx, msg in enumerate(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" or isinstance(msg.get("tool_calls"), list):
            start = idx + 1
    return _last_assistant_text((messages or [])[start:])


def _last_assistant_text(messages: List[Any]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _message_text(msg)
        if text:
            return text
    return ""


def _messages_after_latest_user(messages: List[Any]) -> List[Any]:
    if not messages:
        return []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return messages[idx + 1 :]
    return messages


def _message_fingerprint(message: Any) -> str:
    if not isinstance(message, dict):
        return _string(message)
    role = _string(message.get("role"))
    name = _string(message.get("name"))
    tool_call_id = _string(message.get("tool_call_id"))
    tool_calls = message.get("tool_calls")
    tool_calls_text = ""
    if isinstance(tool_calls, list):
        tool_calls_text = json.dumps(tool_calls, sort_keys=True, ensure_ascii=False, default=str)
    return "\n".join(
        (
            role,
            name,
            tool_call_id,
            _message_text(message),
            tool_calls_text,
        )
    ).strip()


def _current_turn_response_messages(messages: List[Any], prior_messages: List[Any]) -> List[Any]:
    prior = {_message_fingerprint(msg) for msg in prior_messages or [] if _message_fingerprint(msg)}
    filtered = [
        msg
        for msg in messages or []
        if not prior or _message_fingerprint(msg) not in prior
    ]
    return _messages_after_latest_user(filtered)


def _has_visible_agent_output(messages: List[Any], final_response: str) -> bool:
    if _string(final_response):
        return True
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant" and _message_text(msg):
            return True
        if role == "tool" and _tool_result_success(msg.get("content")):
            return True
    return False


def _callback_backend_url(req: AlphartEduChatRequest) -> str:
    return _backend_url_from_req(req)


def _callback_service_token() -> str:
    return _string(
        os.getenv("HERMES_AGENT_TOKEN")
        or os.getenv("ALPHART_AGENT_TOKEN")
        or os.getenv("CANVAS_AGENT_TOKEN")
    )


def _alphart_enabled_toolsets(req: AlphartEduChatRequest) -> List[str]:
    if req.script_only:
        # Canvas uses this mode to draft an Audio node's spoken script. Media tools
        # must be unavailable even when the brief contains words such as "audio".
        return ["skills"]
    if _request_app_scope(req) == "canvas":
        if _canvas_shot_breakdown_intent(req):
            # The product workflow is already preloaded. Excluding the generic
            # skills toolset prevents a model from attempting an optional
            # skill_view lookup instead of the required video analysis call.
            return ["alphart-canvas", "video"]
        return ["alphart-canvas", "skills"]
    return ["alphart-edu", "skills"]


def _post_chat_result_callback(req: AlphartEduChatRequest, response: Dict[str, Any]) -> None:
    backend_url = _callback_backend_url(req)
    if not backend_url:
        print(
            f"[alphart-agent] chat result callback skipped session_id={req.session_id} reason=missing_backend_url",
            flush=True,
        )
        return
    token = _callback_service_token()
    payload = dict(response)
    payload.update(
        {
            "session_id": req.session_id,
            "canvas_id": req.canvas_id,
            "user_id": req.user_id,
            "request_messages": req.messages,
        }
    )
    try:
        resp = requests.post(
            f"{backend_url}/internal/api/v1/agent/chat-results",
            json=payload,
            headers={
                **({"Authorization": f"Bearer {token}"} if token else {}),
                **({"X-Hermes-Agent-Token": token} if token else {}),
            },
            timeout=int(os.getenv("ALPHART_EDU_BACKEND_CALLBACK_TIMEOUT_SECONDS") or os.getenv("CANVAS_BACKEND_CALLBACK_TIMEOUT_SECONDS", "30")),
        )
    except requests.RequestException as exc:
        print(
            f"[alphart-agent] chat result callback failed session_id={req.session_id} error={exc}",
            flush=True,
        )
        return
    preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] chat result callback response session_id={req.session_id} status={resp.status_code} bytes={len(resp.text)} body={preview}",
        flush=True,
    )


def _post_chat_event_callback(req: AlphartEduChatRequest, event: Dict[str, Any]) -> None:
    backend_url = _callback_backend_url(req)
    if not backend_url or not req.session_id or not isinstance(event, dict):
        return
    token = _callback_service_token()
    clean_event = dict(event)
    clean_event.pop("_live_sent", None)
    try:
        requests.post(
            f"{backend_url}/internal/api/v1/agent/events",
            json={
                "session_id": req.session_id,
                "canvas_id": req.canvas_id,
                "user_id": req.user_id,
                "event": clean_event,
            },
            headers={
                **({"Authorization": f"Bearer {token}"} if token else {}),
                **({"X-Hermes-Agent-Token": token} if token else {}),
            },
            timeout=3,
        )
    except requests.RequestException:
        return


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name) or 0)
    return int(getattr(usage, name, 0) or 0)


_TITLE_SYSTEM = (
    "You generate short display names for chat sessions and canvases. "
    "The user content is raw source text, not a request for you to perform. "
    "Name the actual subject, scene, asset, lesson, or task being discussed. "
    "If the source asks to generate or regenerate media, do NOT title it "
    "'request to generate'; title it as the intended subject, for example "
    "'Shanghai Huangpu River Night Scene'. Return only the display name: "
    "no quotes, no markdown, no 'Title:', no 'request', no commentary. "
    "Use the same language as the source when natural. Max 8 words."
)


def _title_prompt(source: str) -> str:
    return (
        "Create a concise session/canvas display name from this source text. "
        "Extract the subject instead of describing the user's request. "
        "Return only the name.\n\n"
        f"{source}"
    )


def _provider_format(provider: str, endpoint: str, model: str = "") -> str:
    """Mirror relay.go textProviderFormat: returns 'anthropic', 'gemini', or 'openai'.

    Model name is the most reliable signal when the provider/endpoint strings are
    generic (e.g. provider='text', endpoint='https://my-proxy.example.com').
    """
    p = provider.lower()
    u = endpoint.lower()
    m = model.lower()
    if "claude" in p or "anthropic" in p or "anthropic.com" in u or m.startswith("claude"):
        return "anthropic"
    if (
        "generativelanguage.googleapis.com" in u
        or ":generatecontent" in u
        or m.startswith("gemini")
        or (not u and ("vertex" in p or "gemini" in p or "google" in p))
    ):
        return "gemini"
    return "openai"


def _model_wire_format(provider: str, model: str) -> str:
    p = provider.lower()
    m = model.lower()
    if "claude" in p or "anthropic" in p or m.startswith("claude") or m.startswith("anthropic."):
        return "anthropic"
    if (
        "gemini" in p
        or "vertex" in p
        or "google" in p
        or m.startswith("gemini")
        or m.startswith("models/gemini")
    ):
        return "gemini"
    if (
        "openai" in p
        or m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("o5")
    ):
        return "openai"
    return ""


def _endpoint_wire_format(endpoint: str) -> str:
    """Infer wire format from endpoint shape.

    OpenAI-compatible gateways commonly expose a generic /v1 base URL while
    serving Claude/Gemini-named models. In that case the endpoint, not the model
    name, determines the request/streaming protocol.
    """
    u = endpoint.lower().rstrip("/")
    if not u:
        return ""
    if "generativelanguage.googleapis.com" in u or ":generatecontent" in u:
        return "gemini"
    if "anthropic.com" in u or u.endswith("/messages"):
        return "anthropic"
    if u.endswith("/v1") or "/v1/" in u or u.endswith("/chat/completions"):
        return "openai"
    return ""


def _text_model_wire_format(provider: str, model: str, config: Dict[str, Any]) -> str:
    """Return the API wire format for a text model.

    Model/provider names are the primary signal for Alphart's internal relay:
    Claude-named models use Anthropic Messages, Gemini-named models use Gemini,
    and GPT/O-named models use OpenAI-compatible chat completions.
    """
    model_format = _model_wire_format(provider, model)
    if model_format:
        return model_format

    explicit = _string(
        config.get("wire_format")
        or config.get("provider_format")
        or config.get("api_format")
        or config.get("format")
    ).lower()
    explicit_aliases = {
        "openai": "openai",
        "openai_compatible": "openai",
        "openai-compatible": "openai",
        "chat_completions": "openai",
        "chat-completions": "openai",
        "anthropic": "anthropic",
        "anthropic_messages": "anthropic",
        "anthropic-messages": "anthropic",
        "claude": "anthropic",
        "gemini": "gemini",
        "google": "gemini",
        "vertex": "gemini",
        "generate_content": "gemini",
        "generate-content": "gemini",
    }
    if explicit in explicit_aliases:
        return explicit_aliases[explicit]

    api_mode = _string(config.get("api_mode")).lower()
    if api_mode in explicit_aliases:
        return explicit_aliases[api_mode]

    endpoint = _endpoint(config)
    endpoint_format = _endpoint_wire_format(endpoint)
    if endpoint_format:
        return endpoint_format

    return _provider_format(provider, endpoint, model)


def _agent_provider_mode_for_wire_format(provider_format: str) -> Tuple[str, str]:
    if provider_format == "anthropic":
        return "anthropic", "anthropic_messages"
    if provider_format == "gemini":
        return "gemini", "chat_completions"
    return "openai", "chat_completions"


def _internal_relay_agent_mode(provider: str, model: str, config: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    """Return agent transport for Alphart's internal relay.

    The internal relay exposes both OpenAI-compatible chat-completions and
    Anthropic-compatible messages and Gemini-compatible generateContent
    surfaces. Claude models must use the official Anthropic SDK transport
    against /internal/messages; Gemini models use Hermes's Gemini native
    adapter against /internal/gemini/v1beta. Routing either through
    /internal/chat/completions loses provider-native semantics.
    """
    upstream_format = _text_model_wire_format(provider, model, config) or "openai"
    if upstream_format == "anthropic":
        return "anthropic", "anthropic_messages", upstream_format, True
    if upstream_format == "gemini":
        return "gemini", "chat_completions", upstream_format, False
    return "openai", "chat_completions", upstream_format, True


def _generate_title_anthropic(endpoint: str, api_key: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    base = (endpoint or "https://api.anthropic.com/v1").rstrip("/")
    if not base.endswith("/messages"):
        base += "/messages"
    body = {
        "model": model,
        "max_tokens": 32,
        "system": _TITLE_SYSTEM,
        "messages": [{"role": "user", "content": _title_prompt(source)}],
    }
    logger.info("title anthropic url=%s model=%s", base, model)
    try:
        resp = requests.post(
            base,
            json=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout as exc:
        logger.error("title anthropic read timeout url=%s timeout=%s: %s", base, timeout, exc)
        raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
    except requests.ConnectionError as exc:
        logger.error("title anthropic connection error url=%s: %s", base, exc)
        raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        body_text = exc.response.text[:500] if exc.response is not None else ""
        logger.error("title anthropic http error url=%s status=%s body=%s", base, status, body_text)
        raise HTTPException(status_code=status, detail=f"Title model error: {exc}") from exc
    data = resp.json()
    content = ""
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            content += block.get("text", "")
    usage = data.get("usage", {})
    return {
        "title": _strip_think_tags(content),
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
    }


def _gemini_chat_url(endpoint: str, model: str) -> str:
    """Mirror geminiChatEndpoint in relay.go: resolve the generateContent URL."""
    if "%s" in endpoint:
        return endpoint % model
    endpoint = endpoint.rstrip("/")
    if ":generatecontent" in endpoint.lower():
        return endpoint
    if "/models/" in endpoint.lower():
        return endpoint + ":generateContent"
    return endpoint + "/models/" + model + ":generateContent"


def _generate_title_gemini(endpoint: str, api_key: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    url = _gemini_chat_url(endpoint, model)
    body = {
        "contents": [{"role": "user", "parts": [{"text": _title_prompt(source)}]}],
        "systemInstruction": {"parts": [{"text": _TITLE_SYSTEM}]},
        "generationConfig": {"maxOutputTokens": 32, "temperature": 0.2},
    }
    # Mirror relay.go callGeminiChat: send both auth styles so the request
    # works with Google AI Studio directly and with local proxies that may
    # require one or the other.
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + (api_key or ""),
        "x-goog-api-key": api_key or "",
    }
    logger.info("title gemini url=%s model=%s", url, model)
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout as exc:
        logger.error("title gemini read timeout url=%s timeout=%s: %s", url, timeout, exc)
        raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
    except requests.ConnectionError as exc:
        logger.error("title gemini connection error url=%s: %s", url, exc)
        raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        body_text = exc.response.text[:500] if exc.response is not None else ""
        logger.error("title gemini http error url=%s status=%s body=%s", url, status, body_text)
        raise HTTPException(status_code=status, detail=f"Title model error: {exc}") from exc
    data = resp.json()
    content = ""
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            content += part.get("text", "")
    usage = data.get("usageMetadata", {})
    return {
        "title": _strip_think_tags(content),
        "prompt_tokens": int(usage.get("promptTokenCount") or 0),
        "completion_tokens": int(usage.get("candidatesTokenCount") or 0),
        "total_tokens": int(usage.get("totalTokenCount") or 0),
    }


def _generate_title_direct(provider: str, endpoint: str, api_key: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    fmt = _provider_format(provider, endpoint, model)
    logger.info("title provider=%s model=%s endpoint=%s format=%s", provider, model, endpoint, fmt)
    if fmt == "anthropic":
        return _generate_title_anthropic(endpoint, api_key, model, source, config)
    if fmt == "gemini":
        return _generate_title_gemini(endpoint, api_key, model, source, config)
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    client = OpenAI(api_key=api_key, base_url=endpoint, timeout=timeout)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM},
                {"role": "user", "content": _title_prompt(source)},
            ],
            max_tokens=32,
            temperature=0.2,
        )
    except OpenAITimeoutError as exc:
        logger.error("title openai read timeout endpoint=%s model=%s timeout=%s: %s", endpoint, model, timeout, exc)
        raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
    except OpenAIConnectionError as exc:
        logger.error("title openai connection error endpoint=%s model=%s: %s", endpoint, model, exc)
        raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
    except OpenAIStatusError as exc:
        logger.error("title openai status error endpoint=%s model=%s status=%s: %s", endpoint, model, exc.status_code, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=f"Title model error: {exc.message}") from exc
    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    content = _strip_think_tags(content)
    usage = getattr(response, "usage", None)
    return {
        "title": content,
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _generate_title_relay(req: AlphartEduTitleRequest, provider: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    endpoint = _internal_relay_base_url(req)
    headers = _internal_relay_headers(req)
    wire_format = _text_model_wire_format(provider, model, config)
    if wire_format == "anthropic":
        url = endpoint.rstrip("/") + "/messages"
        relay_headers = dict(headers)
        relay_headers.update(
            {
                "authorization": "Bearer " + _internal_relay_api_key(),
                "x-api-key": _internal_relay_api_key(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
        body = {
            "model": model,
            "max_tokens": 32,
            "system": _TITLE_SYSTEM,
            "messages": [{"role": "user", "content": _title_prompt(source)}],
        }
        logger.info("title relay anthropic provider=%s model=%s endpoint=%s org_no=%s", provider, model, url, headers.get("X-Org-No", ""))
        try:
            resp = requests.post(url, json=body, headers=relay_headers, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            logger.error("title relay anthropic read timeout endpoint=%s model=%s timeout=%s: %s", url, model, timeout, exc)
            raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
        except requests.ConnectionError as exc:
            logger.error("title relay anthropic connection error endpoint=%s model=%s: %s", url, model, exc)
            raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 502
            body_text = exc.response.text[:500] if exc.response is not None else ""
            logger.error("title relay anthropic http error endpoint=%s model=%s status=%s body=%s", url, model, status, body_text)
            raise HTTPException(status_code=status, detail=f"Title model error: {body_text or exc}") from exc
        data = resp.json()
        content = ""
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                content += block.get("text", "")
        usage = data.get("usage", {})
        return {
            "title": _strip_think_tags(content),
            "model": model,
            "provider": provider,
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
        }

    if wire_format == "gemini":
        safe_model = quote(model.lstrip("/").removeprefix("models/"), safe="/")
        url = endpoint.rstrip("/") + f"/gemini/v1beta/models/{safe_model}:generateContent"
        relay_headers = dict(headers)
        relay_headers.update(
            {
                "authorization": "Bearer " + _internal_relay_api_key(),
                "x-api-key": _internal_relay_api_key(),
                "content-type": "application/json",
            }
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": _title_prompt(source)}]}],
            "systemInstruction": {"parts": [{"text": _TITLE_SYSTEM}]},
            "generationConfig": {"maxOutputTokens": 32, "temperature": 0.2},
        }
        logger.info("title relay gemini provider=%s model=%s endpoint=%s org_no=%s", provider, model, url, headers.get("X-Org-No", ""))
        try:
            resp = requests.post(url, json=body, headers=relay_headers, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            logger.error("title relay gemini read timeout endpoint=%s model=%s timeout=%s: %s", url, model, timeout, exc)
            raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
        except requests.ConnectionError as exc:
            logger.error("title relay gemini connection error endpoint=%s model=%s: %s", url, model, exc)
            raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 502
            body_text = exc.response.text[:500] if exc.response is not None else ""
            logger.error("title relay gemini http error endpoint=%s model=%s status=%s body=%s", url, model, status, body_text)
            raise HTTPException(status_code=status, detail=f"Title model error: {body_text or exc}") from exc
        data = resp.json()
        content = ""
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part, dict):
                    content += _string(part.get("text"))
        usage = data.get("usageMetadata", {})
        return {
            "title": _strip_think_tags(content),
            "model": model,
            "provider": provider,
            "prompt_tokens": int(usage.get("promptTokenCount") or 0),
            "completion_tokens": int(usage.get("candidatesTokenCount") or 0),
            "total_tokens": int(usage.get("totalTokenCount") or 0),
        }

    client = OpenAI(api_key=_internal_relay_api_key(), base_url=endpoint, timeout=timeout)
    logger.info("title relay provider=%s model=%s endpoint=%s wire_format=%s org_no=%s", provider, model, endpoint, wire_format, headers.get("X-Org-No", ""))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TITLE_SYSTEM},
                {"role": "user", "content": _title_prompt(source)},
            ],
            max_tokens=32,
            temperature=0.2,
            extra_headers=headers,
        )
    except OpenAITimeoutError as exc:
        logger.error("title relay read timeout endpoint=%s model=%s timeout=%s: %s", endpoint, model, timeout, exc)
        raise HTTPException(status_code=504, detail=f"Title model timed out after {timeout}s") from exc
    except OpenAIConnectionError as exc:
        logger.error("title relay connection error endpoint=%s model=%s: %s", endpoint, model, exc)
        raise HTTPException(status_code=502, detail=f"Title model connection error: {exc}") from exc
    except OpenAIStatusError as exc:
        logger.error("title relay status error endpoint=%s model=%s status=%s: %s", endpoint, model, exc.status_code, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=f"Title model error: {exc.message}") from exc
    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return {
        "title": _strip_think_tags(content),
        "model": model,
        "provider": provider,
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _generate_title_agent(req: AlphartEduTitleRequest, provider: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    agent_provider = provider
    api_mode = _string(config.get("api_mode")) or "chat_completions"
    stream_enabled = True
    relay_headers: Dict[str, str] = {}
    if _use_internal_relay(req):
        provider_format = _text_model_wire_format(provider, model, config)
        endpoint = _internal_relay_gemini_base_url(req) if provider_format == "gemini" else _internal_relay_base_url(req)
        api_key = _internal_relay_api_key()
        relay_headers = _internal_relay_headers(req)
        request_overrides = {"extra_headers": relay_headers}
        agent_provider, api_mode, provider_format, stream_enabled = _internal_relay_agent_mode(provider, model, config)
    else:
        endpoint = _endpoint(config)
        api_key = _api_key(config)
        request_overrides = None
    if not endpoint or not api_key:
        raise HTTPException(status_code=400, detail="title model endpoint/api key is not configured")

    if not _use_internal_relay(req) and _provider_format(provider, endpoint, model) == "anthropic":
        api_mode = "chat_completions"
    agent = AIAgent(
        base_url=endpoint,
        api_key=api_key,
        provider=agent_provider,
        api_mode=api_mode,
        model=model,
        enabled_toolsets=[],
        disabled_toolsets=["alphart-edu", "alphart-canvas", "skills"],
        max_iterations=1,
        max_tokens=64,
        quiet_mode=True,
        session_id=None,
        skip_context_files=True,
        skip_memory=True,
        platform="alphart",
        user_id=req.user_id or None,
        request_overrides=request_overrides,
    )
    if api_mode == "anthropic_messages":
        _install_internal_relay_anthropic_headers(agent, relay_headers)
    if not stream_enabled:
        agent._disable_streaming = True
    try:
        result = agent.run_conversation(
            _title_prompt(source),
            system_message=_TITLE_SYSTEM,
            conversation_history=[],
            task_id=None,
        )
    except Exception as exc:
        logger.warning("title agent error provider=%r model=%r: %s", provider, model, exc)
        raise HTTPException(status_code=502, detail=f"Title agent error: {exc}") from exc

    return {
        "title": _strip_think_tags(_string(result.get("final_response"))),
        "model": result.get("model") or model,
        "provider": result.get("provider") or provider,
        "prompt_tokens": int(result.get("prompt_tokens") or result.get("input_tokens") or 0),
        "completion_tokens": int(result.get("completion_tokens") or result.get("output_tokens") or 0),
        "total_tokens": int(result.get("total_tokens") or 0),
    }


def _agent_max_iterations(config: Dict[str, Any]) -> int:
    raw = config.get("max_iterations") or os.getenv("HERMES_AGENT_MAX_ITERATIONS") or 30
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 30
    return max(value, 3)


def _tool_call_name(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function")
    if isinstance(function, dict):
        return _string(function.get("name"))
    return _string(tool_call.get("name"))


def _tool_call_arguments(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return "{}"
    function = tool_call.get("function")
    if isinstance(function, dict):
        raw = function.get("arguments")
    else:
        raw = tool_call.get("arguments")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return "{}"


def _events_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    hidden_tool_call_ids = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tool_call in msg.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = _string(tool_call.get("id"))
                tool_name = _tool_call_name(tool_call)
                if not tool_call_id or not tool_name:
                    continue
                if _is_internal_tool_view_name(tool_name):
                    hidden_tool_call_ids.add(tool_call_id)
                    continue
                events.append({"type": "tool_call", "id": tool_call_id, "name": tool_name})
                args = _tool_call_arguments(tool_call)
                if args and args != "{}":
                    events.append({"type": "tool_call_arguments", "id": tool_call_id, "text": args})
        elif msg.get("role") == "tool":
            tool_call_id = _string(msg.get("tool_call_id"))
            if tool_call_id in hidden_tool_call_ids:
                continue
            if _is_internal_tool_view_name(msg.get("name") or msg.get("tool_name")):
                continue
            if tool_call_id:
                events.append({"type": "tool_call_result", "id": tool_call_id, "message": msg})
    return events


@app.get("/health")
@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/v1/agent/chats")
def chat(req: AlphartEduChatRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_auth(authorization)

    candidates = _text_model_candidates(req)
    if not candidates:
        raise HTTPException(status_code=400, detail="text model provider/model is required")

    primary_text_model = candidates[0]
    raw_messages = [msg for msg in req.messages if isinstance(msg, dict)]
    messages = _fix_chat_history(_filter_image_content(raw_messages, primary_text_model))
    last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), -1)
    if last_user_index >= 0:
        latest_user = messages[last_user_index]
        user_message = _message_text(latest_user)
        user_content = _message_content(latest_user)
        conversation_history = messages[:last_user_index]
    else:
        latest_user = messages[-1] if messages else {}
        user_message = _message_text(latest_user) if messages else ""
        user_content = _message_content(latest_user) if messages else user_message
        conversation_history = messages[:-1] if messages else []
    if not user_message:
        raise HTTPException(status_code=400, detail="user message is required")
    is_storybook_request = _storybook_intent(user_message)
    if is_storybook_request:
        candidates = _require_openai_text_model_candidates(req, candidates, "storybook", exclude_small_models=True)
        primary_text_model = candidates[0]
    is_game_request = _game_intent(user_message)
    if is_game_request and not is_storybook_request:
        candidates = _require_openai_text_model_candidates(req, candidates, "game generation", exclude_small_models=True)
        primary_text_model = candidates[0]
    canvas_input_images = list(req.input_images or []) if _request_app_scope(req) == "canvas" else []
    canvas_input_audio = list(req.input_audio or []) if _request_app_scope(req) == "canvas" else []
    input_images = canvas_input_images or _input_images_from_text(user_message)
    latest_generated_image = _latest_generated_image_ref(conversation_history)
    if not input_images and latest_generated_image and _media_intent(user_message, has_image_context=True) == "image":
        input_images = [latest_generated_image]
    model_user_message: Any = user_message
    if _model_supports_vision(primary_text_model) and _has_media_content(user_content):
        prepared_content = _prepare_chat_content_for_model(req, user_content)
        if isinstance(prepared_content, list) and prepared_content:
            model_user_message = prepared_content

    multimodal_model = dict(req.multimodal_model) if isinstance(req.multimodal_model, dict) else {}
    multimodal_config = _provider_config_for_domain(req.model_configs, "multimodal", multimodal_model)
    context = {
        "session_id": req.session_id,
        "canvas_id": req.canvas_id,
        "canvas_item_id": req.canvas_item_id,
        "selected_canvas_item_id": req.selected_canvas_item_id,
        "selected_canvas_item_type": req.selected_canvas_item_type,
        "canvas_item_type": req.canvas_item_type,
        "force_media_intent": req.force_media_intent,
        "canvas_prompt_context": req.canvas_prompt_context,
        "image_model": req.image_model,
        "video_model": req.video_model,
        "audio_model": req.audio_model,
        "user_id": req.user_id,
        "user_uuid": req.user_uuid,
        "storage_prefix": req.storage_prefix,
        "org_no": req.org_no,
        "auth_token": req.auth_token,
        "backend_url": _backend_url_from_req(req),
        "app_scope": _request_app_scope(req),
        "user_message": user_message,
        "tool_list": req.tool_list,
        "input_images": input_images,
        "input_audio": canvas_input_audio,
        "reference_item_ids": list(req.reference_item_ids or []),
        "duration_seconds": int(req.duration_seconds or 0),
        # Canvas keeps audio duration independent from video duration. Edu
        # requests intentionally receive zero here and retain their legacy path.
        "audio_duration_seconds": int(req.audio_duration_seconds or 0) if _request_app_scope(req) == "canvas" else 0,
        "aspect_ratio": req.aspect_ratio,
        "resolution": req.resolution,
        "image_quality": req.image_quality,
        "image_resolution": req.image_resolution,
        "image_aspect_ratio": req.image_aspect_ratio,
        "generate_audio": bool(req.generate_audio),
        "video_caption_script": req.video_caption_script,
        "script_only": bool(req.script_only),
        "approved_audio_script": req.approved_audio_script,
        "audio_language_type": _normalize_audio_language_type(req.audio_language_type) or _ui_audio_language_type(req.ui_language),
        "ui_language": req.ui_language,
    }
    if _request_app_scope(req) == "canvas":
        context["multimodal_runtime"] = {
            "provider": _string(multimodal_model.get("provider")),
            "model": _string(multimodal_model.get("model")),
            "base_url": _endpoint(multimodal_config),
            "api_key": _api_key(multimodal_config),
            "timeout": multimodal_config.get("timeout"),
        }

    events: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {}
    last_provider = ""
    last_model = ""
    for candidate in candidates:
        provider = _string(candidate.get("provider"))
        model = _string(candidate.get("model"))
        config = _provider_config_for(req.model_configs, candidate)
        endpoint = _endpoint(config)
        api_key = _api_key(config)
        provider_format = _text_model_wire_format(provider, model, config)
        stream_enabled = provider_format == "openai"
        relay_headers: Dict[str, str] = {}
        agent_provider = provider
        agent_api_mode = _string(config.get("api_mode")) or "chat_completions"
        if _use_internal_relay(req):
            endpoint = _internal_relay_gemini_base_url(req) if provider_format == "gemini" else _internal_relay_base_url(req)
            api_key = _internal_relay_api_key()
            relay_headers = _internal_relay_headers(req)
            agent_provider, agent_api_mode, provider_format, stream_enabled = _internal_relay_agent_mode(provider, model, config)
            print(
                f"[alphart-agent] internal relay text model session_id={req.session_id} "
                f"provider={provider} agent_provider={agent_provider} model={model} wire_format={provider_format} "
                f"stream={stream_enabled}",
                flush=True,
            )
        if not provider or not model or not endpoint or not api_key:
            print(
                f"[alphart-agent] skipping unconfigured text model session_id={req.session_id} provider={provider} model={model}",
                flush=True,
            )
            continue
        last_provider, last_model = provider, model
        retry_count = max(1, _int(candidate.get("retry"), 1))
        for attempt in range(1, retry_count + 1):
            attempt_events: List[Dict[str, Any]] = []

            def on_delta(*args: Any, **kwargs: Any) -> None:
                text = ""
                if args:
                    text = _string(args[0])
                if not text:
                    text = _string(kwargs.get("delta") or kwargs.get("text"))
                if text:
                    event = {"type": "delta", "text": text, "_live_sent": True}
                    attempt_events.append(event)
                    _post_chat_event_callback(req, event)

            def on_status(*args: Any, **kwargs: Any) -> None:
                message = _string(args[1] if len(args) > 1 else (args[0] if args else kwargs.get("message")))
                if message:
                    event = {"type": "status", "message": message, "_live_sent": True}
                    attempt_events.append(event)
                    _post_chat_event_callback(req, event)

            def emit_live_event(event: Dict[str, Any]) -> None:
                if not event:
                    return
                event["_live_sent"] = True
                attempt_events.append(event)
                _post_chat_event_callback(req, event)

            def on_interim_assistant(text: str, **kwargs: Any) -> None:
                if kwargs.get("already_streamed"):
                    return
                visible = _string(text).strip()
                if not visible:
                    return
                emit_live_event({"type": "delta", "text": visible + "\n\n"})

            def on_tool_start(tool_call_id: str, name: str, args: Any) -> None:
                tool_call_id = _string(tool_call_id)
                name = _string(name)
                if not tool_call_id or not name or _is_internal_tool_view_name(name):
                    return
                emit_live_event({"type": "tool_call", "id": tool_call_id, "name": name})
                if isinstance(args, str):
                    args_text = args
                else:
                    try:
                        args_text = json.dumps(args or {}, ensure_ascii=False)
                    except TypeError:
                        args_text = "{}"
                if args_text and args_text != "{}":
                    emit_live_event({"type": "tool_call_arguments", "id": tool_call_id, "text": args_text})

            def on_tool_complete(tool_call_id: str, name: str, args: Any, result_text: Any) -> None:
                tool_call_id = _string(tool_call_id)
                name = _string(name)
                if not tool_call_id or not name or _is_internal_tool_view_name(name):
                    return
                content = result_text if isinstance(result_text, str) else _string(result_text)
                if not content:
                    return
                emit_live_event({
                    "type": "tool_call_result",
                    "id": tool_call_id,
                    "message": {
                        "role": "tool",
                        "name": name,
                        "tool_name": name,
                        "tool_call_id": tool_call_id,
                        "content": content,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                })

            with alphart_context(context):
                try:
                    agent = AIAgent(
                        base_url=endpoint,
                        api_key=api_key,
                        provider=agent_provider,
                        api_mode=agent_api_mode,
                        model=model,
                        enabled_toolsets=_alphart_enabled_toolsets(req),
                        max_iterations=_agent_max_iterations(config),
                        max_tokens=_agent_max_tokens(config, is_game=is_game_request),
                        quiet_mode=True,
                        session_id=req.session_id or None,
                        stream_delta_callback=on_delta if stream_enabled else None,
                        interim_assistant_callback=on_interim_assistant,
                        tool_start_callback=on_tool_start,
                        tool_complete_callback=on_tool_complete,
                        status_callback=on_status,
                        platform="alphart",
                        user_id=req.user_id or req.user_uuid or None,
                        chat_id=req.session_id or None,
                        skip_memory=True,
                        skip_context_files=True,
                        request_overrides={"extra_headers": relay_headers} if relay_headers else None,
                        reasoning_config=_canvas_reasoning_config(req),
                    )
                    if agent_api_mode == "anthropic_messages":
                        _install_internal_relay_anthropic_headers(agent, relay_headers)
                    if not stream_enabled:
                        agent._disable_streaming = True
                    result = agent.run_conversation(
                        model_user_message,
                        system_message=_system_prompt(req),
                        conversation_history=conversation_history,
                        task_id=req.session_id or None,
                        persist_user_message=user_message,
                    )
                except Exception as exc:
                    print(
                        f"[alphart-agent] chat model failed session_id={req.session_id} provider={provider} "
                        f"model={model} attempt={attempt}/{retry_count} error={exc}",
                        flush=True,
                    )
                    result = {"failed": True, "error": str(exc), "model": model, "provider": provider}

            if not result.get("failed"):
                events = attempt_events
                break
            print(
                f"[alphart-agent] chat model attempt failed session_id={req.session_id} provider={provider} "
                f"model={model} attempt={attempt}/{retry_count} error={result.get('error')}",
                flush=True,
            )
            events = attempt_events
        if not result.get("failed"):
            break

    model_failed = bool(result.get("failed"))
    model_error = _string(result.get("error"))
    if model_failed:
        visible_error = INSUFFICIENT_CREDITS_MESSAGE if _is_insufficient_credits_error(model_error) else SYSTEM_BUSY_MESSAGE
        result = {
            "messages": [{"role": "assistant", "content": visible_error}],
            "final_response": visible_error,
            "model": result.get("model") or last_model,
            "provider": result.get("provider") or last_provider,
            "failed": True,
            "error": visible_error,
        }

    raw_result_messages = result.get("messages") or []
    response_messages = _public_messages(raw_result_messages)
    response_messages = _current_turn_response_messages(response_messages, conversation_history)
    current_turn_messages = _messages_after_latest_user(response_messages)
    current_media_attempted = _generation_tool_attempted(current_turn_messages)
    current_media_failed = _generation_tool_effectively_failed(current_turn_messages)
    current_game_failed = _game_tool_failed(current_turn_messages)
    current_tool_failed = current_media_failed or current_game_failed
    canvas_tool_error = _generation_tool_error(current_turn_messages) if _request_app_scope(req) == "canvas" else ""
    reference_image_generation = (
        bool(input_images)
        and not current_media_attempted
        and not _storybook_intent(user_message)
        and not _storybook_page_update_intent(user_message)
        and _media_intent(user_message, has_image_context=True) == "image"
    )
    if reference_image_generation:
        if _generation_tool_completed(current_turn_messages, "image"):
            response_messages = current_turn_messages
        else:
            if response_messages:
                print(
                    f"[alphart-agent] discarding non-tool reference image response session_id={req.session_id} message_count={len(response_messages)}",
                    flush=True,
                )
            response_messages = []
    final_response = _string(result.get("final_response"))
    if current_tool_failed:
        final_response = canvas_tool_error or SYSTEM_BUSY_MESSAGE
    if reference_image_generation and not _generation_tool_completed(response_messages, "image"):
        final_response = ""
    if not final_response:
        if reference_image_generation:
            final_response = _last_assistant_text(response_messages)
        else:
            final_response = (
                _last_assistant_text_after_last_tool(response_messages)
                or _storybook_completion_text(response_messages)
                or _last_assistant_text(response_messages)
                or _last_assistant_text(raw_result_messages)
            )
    user_audio_urls = _audio_urls_from_content(user_content)
    if not model_failed:
        with alphart_context(context):
            current_turn_messages = _messages_after_latest_user(response_messages)
            forced_messages = []
            if user_audio_urls and not _generation_tool_attempted(current_turn_messages, "audio"):
                forced_messages = _forced_audio_to_media_pipeline(
                    user_audio_urls,
                    user_message,
                    response_messages,
                    current_turn_messages,
                    input_images=input_images,
                )
            if (
                not forced_messages
                and _request_app_scope(req) != "canvas"
                and _storybook_intent(user_message)
                and not _storybook_tool_attempted(current_turn_messages)
            ):
                forced_messages = _forced_storybook_tool_messages(
                    user_message,
                    input_images=input_images,
                    audio_language_type=context.get("audio_language_type", ""),
                )
            if (
                not forced_messages
                and _request_app_scope(req) != "canvas"
                and _storybook_page_update_intent(user_message)
                and not _storybook_tool_attempted(current_turn_messages)
            ):
                forced_messages = _forced_storybook_page_update_messages(
                    user_message,
                    input_images=input_images,
                )
            canvas_item_type = _string(req.canvas_item_type).strip().lower()
            requested_media_intent = _string(req.force_media_intent).strip().lower()
            canvas_forced_intent = ""
            if _request_app_scope(req) == "canvas" and not _canvas_shot_breakdown_intent(req):
                requested_or_selected_intent = requested_media_intent or canvas_item_type
                if requested_or_selected_intent in {"image", "video", "audio"}:
                    canvas_forced_intent = requested_or_selected_intent
            if canvas_forced_intent:
                print(
                    "[alphart-agent] canvas media request "
                    f"session_id={req.session_id} item_type={canvas_item_type} "
                    f"force_media_intent={requested_media_intent or canvas_forced_intent} "
                    f"input_images={len(input_images)} input_audio={len(canvas_input_audio)}",
                    flush=True,
                )
            if not forced_messages and not current_media_attempted and not req.script_only:
                forced_messages = _forced_media_tool_messages(
                    user_message,
                    response_messages,
                    current_turn_messages,
                    has_image_context=bool(input_images),
                    input_images=input_images,
                    approved_audio_script=req.approved_audio_script,
                    # A Canvas composer is bound to a selected media node. Its type is
                    # a stronger signal than whether the user's descriptive prompt
                    # happens to contain a word such as "generate".
                    forced_intent="audio" if req.approved_audio_script else canvas_forced_intent,
                )
            response_messages.extend(forced_messages)
    current_turn_messages = _messages_after_latest_user(response_messages)
    current_media_failed = current_media_failed or _generation_tool_effectively_failed(current_turn_messages)
    current_game_failed = current_game_failed or _game_tool_failed(current_turn_messages)
    current_tool_failed = current_media_failed or current_game_failed
    canvas_tool_error = _generation_tool_error(current_turn_messages) if _request_app_scope(req) == "canvas" else ""
    if current_tool_failed:
        final_response = canvas_tool_error or SYSTEM_BUSY_MESSAGE
    if not final_response or (
        _storybook_tool_attempted(current_turn_messages)
        and final_response == _last_assistant_text(response_messages)
        and not _last_assistant_text_after_last_tool(current_turn_messages)
    ):
        final_response = _storybook_completion_text(current_turn_messages) or final_response
    response_messages = _append_visible_generated_media(response_messages, current_turn_messages)
    response_messages = _sanitize_assistant_media_url_text(response_messages)
    final_response = _strip_media_urls_from_text(final_response)
    if reference_image_generation:
        final_response = _last_assistant_text(response_messages)
    empty_result_error = ""
    if not _has_visible_agent_output(response_messages, final_response):
        empty_result_error = "Hermes agent completed without returning a response."
        response_messages.append({"role": "assistant", "content": empty_result_error})
        final_response = empty_result_error
    events = [*_events_from_messages(response_messages), *events]
    raw_count = len(raw_result_messages) if isinstance(raw_result_messages, list) else 0
    print(
        "[alphart-agent] chat result "
        f"session_id={req.session_id} raw_messages={raw_count} "
        f"public_messages={len(response_messages)} final_response_len={len(final_response)} "
        f"failed={bool(result.get('failed') or current_tool_failed or empty_result_error)} "
        f"error={empty_result_error or ((canvas_tool_error or SYSTEM_BUSY_MESSAGE) if current_tool_failed else '')}",
        flush=True,
    )
    response = {
        "status": "ok",
        "final_response": final_response,
        "messages": response_messages,
        "events": events,
        "model": result.get("model") or last_model,
        "provider": result.get("provider") or last_provider,
        "prompt_tokens": result.get("prompt_tokens") or 0,
        "completion_tokens": result.get("completion_tokens") or 0,
        "total_tokens": result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "interrupted": bool(result.get("interrupted") or current_tool_failed),
        "failed": bool(result.get("failed") or current_tool_failed or empty_result_error),
        "error": empty_result_error or ((canvas_tool_error or SYSTEM_BUSY_MESSAGE) if current_tool_failed else _string(result.get("error"))),
    }
    _post_chat_result_callback(req, response)
    return response


@app.post("/api/v1/agent/titles")
def title(req: AlphartEduTitleRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_auth(authorization)

    candidates = _text_model_candidates(req)
    if candidates:
        req.text_model = candidates[0]
    provider = _string(req.text_model.get("provider"))
    model = _string(req.text_model.get("model"))
    config = _provider_config(req)
    endpoint = _endpoint(config)
    api_key = _api_key(config)
    config_providers = list(req.model_configs.keys()) if isinstance(req.model_configs, dict) else []
    logger.info(
        "title request provider=%r model=%r endpoint=%r has_key=%s config_providers=%s",
        provider, model, endpoint, bool(api_key), config_providers,
    )
    if not provider or not model:
        logger.warning("title: primary text_model missing provider/model provider=%r model=%r", provider, model)

    source = "\n".join(_message_text(msg) for msg in req.messages if isinstance(msg, dict)).strip()
    if not source:
        raise HTTPException(status_code=400, detail="title source is required")

    # Build ordered candidate list: primary first, then remaining text_models
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    def _add_candidate(m: Dict[str, Any]) -> None:
        p = _string(m.get("provider"))
        mo = _string(m.get("model"))
        if p and mo and (p, mo) not in seen:
            seen.add((p, mo))
            candidates.append(m)
    _add_candidate(req.text_model)
    for m in req.text_models:
        if isinstance(m, dict):
            _add_candidate(m)

    last_exc: Optional[HTTPException] = None
    result: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        cand_provider = _string(candidate.get("provider"))
        cand_model = _string(candidate.get("model"))
        cand_config = _provider_config_for(req.model_configs, candidate)
        cand_endpoint = _endpoint(cand_config)
        cand_key = _api_key(cand_config)
        if not cand_provider or not cand_model:
            continue
        try:
            if not _use_internal_relay(req) and (not cand_endpoint or not cand_key):
                continue
            if _use_internal_relay(req):
                result = _generate_title_relay(req, cand_provider, cand_model, source, cand_config)
            else:
                result = _generate_title_direct(cand_provider, cand_endpoint, cand_key, cand_model, source, cand_config)
            break
        except HTTPException as exc:
            logger.warning("title failed provider=%r model=%r status=%s: %s", cand_provider, cand_model, exc.status_code, exc.detail)
            last_exc = exc
        except Exception as exc:
            logger.warning("title error provider=%r model=%r: %s", cand_provider, cand_model, exc)
            last_exc = HTTPException(status_code=502, detail=str(exc))
    if result is None:
        if last_exc is not None:
            raise last_exc
        raise HTTPException(status_code=400, detail="no usable text model configured for title generation")

    raw_title = _string(result.get("title"))
    cleaned = _clean_llm_title(raw_title)
    return {
        "status": "ok",
        "title": cleaned,
        "model": result.get("model") or model,
        "provider": result.get("provider") or provider,
        "prompt_tokens": result.get("prompt_tokens") or 0,
        "completion_tokens": result.get("completion_tokens") or 0,
        "total_tokens": result.get("total_tokens") or 0,
    }


def main() -> None:
    import uvicorn

    host = os.getenv("HERMES_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("HERMES_AGENT_PORT", "58088"))
    uvicorn.run("alphart_agent_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
