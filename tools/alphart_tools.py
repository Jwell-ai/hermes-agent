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
import json
import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

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


def _tool_error(message: str, code: str = "") -> str:
    payload = {"success": False, "error": message}
    if code:
        payload["code"] = code
    return json.dumps(payload, ensure_ascii=False)


def _system_busy_tool_error() -> str:
    return "generate fail"


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


def _log_model_value(value: Any) -> str:
    text = str(value or "").strip()
    return text or "backend-selected"


def _backend_url() -> str:
    context_url = str(_ctx().get("backend_url") or "").strip()
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


def _internal_api_url(path: str) -> str:
    backend_url = _backend_url()
    if not backend_url:
        return ""
    return f"{backend_url}/internal/api/v1/{path.lstrip('/')}"


def _backend_tool_timeout(default: int = 900) -> int:
    raw = (
        os.getenv("ALPHART_BACKEND_TOOL_TIMEOUT_SECONDS")
        or os.getenv("ALPHART_EDU_BACKEND_TOOL_TIMEOUT_SECONDS")
        or os.getenv("CANVAS_BACKEND_TOOL_TIMEOUT_SECONDS")
        or str(default)
    )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


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


def _handle_canvas_create_node(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if not args.get("canvas_id"):
        args["canvas_id"] = _ctx().get("canvas_id")
    if not args.get("item_type") and args.get("type"):
        args["item_type"] = args.get("type")
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
    return json.dumps({"status": "success", "result": decoded}, ensure_ascii=False)


def _handle_canvas_update_node(args: Dict[str, Any], **_: Any) -> str:
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


def _handle_alphart_generate_image(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        if _ctx().get("aspect_ratio"):
            args["aspect_ratio"] = _ctx().get("aspect_ratio")
        if _ctx().get("image_quality"):
            args["quality"] = _ctx().get("image_quality")
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
    relay_url = _internal_relay_url("images/generations" if is_canvas or not args.get("input_images") else "images/edits")
    if not relay_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    payload = {
        "model": args.get("model"),
        "provider": args.get("provider"),
        "prompt": args.get("prompt"),
        "aspect_ratio": args.get("aspect_ratio"),
        "quality": args.get("quality"),
        "session_id": _ctx().get("session_id"),
        "canvas_id": _ctx().get("canvas_id"),
        "canvas_item_id": args.get("canvas_item_id") or args.get("item_id") or args.get("node_id") or _ctx().get("canvas_item_id"),
    }
    if args.get("quantity"):
        payload["n"] = args.get("quantity")
    if args.get("input_images"):
        payload["images"] = args.get("input_images")
    print(
        f"[alphart-agent] calling internal relay image session_id={_ctx().get('session_id')} url={relay_url}",
        flush=True,
    )
    try:
        resp = requests.post(
            relay_url,
            json=payload,
            headers=_internal_relay_headers(),
            timeout=_backend_tool_timeout(),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Alphart relay request failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] internal relay image response status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
        flush=True,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
            detail = ""
            if isinstance(decoded, dict):
                error = decoded.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("detail") or "")
                detail = detail or str(decoded.get("detail") or decoded.get("message") or "")
            detail = " ".join((detail or f"relay returned HTTP {resp.status_code}").split())[:500]
            return _tool_error(f"Canvas image relay failed (HTTP {resp.status_code}): {detail}")
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
    result = {
        "type": "image",
        "provider": asset.get("provider") or args.get("provider"),
        "model": asset.get("model") or args.get("model"),
        "url": asset.get("url"),
        "mime_type": asset.get("mime_type"),
        "width": asset.get("width"),
        "height": asset.get("height"),
        "filename": asset.get("filename"),
        "s3_object_name": asset.get("s3_object_name"),
        "usage": asset.get("usage"),
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _handle_alphart_generate_video(args: Dict[str, Any], **kwargs: Any) -> str:
    args = dict(args or {})
    is_canvas = str(_ctx().get("app_scope") or "").strip().lower() == "canvas"
    if not str(args.get("prompt") or "").strip() and is_canvas:
        args["prompt"] = str(_ctx().get("canvas_prompt_context") or _ctx().get("user_message") or "").strip()
    if is_canvas and _ctx().get("duration_seconds"):
        args["duration"] = _ctx().get("duration_seconds")
    if is_canvas and _ctx().get("aspect_ratio"):
        args["aspect_ratio"] = _ctx().get("aspect_ratio")
    if is_canvas and _ctx().get("resolution"):
        args["resolution"] = _ctx().get("resolution")
    if args.get("duration_seconds") and not args.get("duration"):
        args["duration"] = args.get("duration_seconds")
    if args.get("image_url") and not args.get("input_images"):
        args["input_images"] = [args.get("image_url")]
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
        # Connected Canvas nodes are the user-selected keyframes. Keep them
        # ahead of any model-suggested images so an audio reference can never
        # reach the video relay without its visual counterpart.
        canvas_images = [entry for entry in (_ctx().get("input_images") or []) if entry]
        requested_images = [entry for entry in (args.get("input_images") or []) if entry]
        merged_images = list(canvas_images)
        known_images = {
            str(entry.get("s3_object_name") or entry.get("object_key") or entry.get("url") or entry)
            if isinstance(entry, dict)
            else str(entry)
            for entry in canvas_images
        }
        for entry in requested_images:
            identity = (
                str(entry.get("s3_object_name") or entry.get("object_key") or entry.get("url") or entry)
                if isinstance(entry, dict)
                else str(entry)
            )
            if identity and identity not in known_images:
                merged_images.append(entry)
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
        # Canvas owns captions and voiceover separately. Seedance should still
        # create the ambient track whenever no BGM/soundtrack is attached,
        # regardless of a model-supplied tool argument.
        args["generate_audio"] = True
    elif "generate_audio" not in args:
        args["generate_audio"] = bool(_ctx().get("generate_audio"))
    tool = _pick_tool("video", args)
    _set_tool_defaults(args, tool)
    args.setdefault("wait", False)
    relay_url = _internal_relay_url("videos")
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
        "canvas_item_id": args.get("canvas_item_id") or args.get("item_id") or args.get("node_id") or _ctx().get("canvas_item_id"),
        "generate_audio": bool(args.get("generate_audio")),
        "caption_script": _ctx().get("video_caption_script"),
        "tool_call_id": kwargs.get("tool_call_id") or args.get("tool_call_id"),
    }
    if str(_ctx().get("app_scope") or "").strip().lower() == "canvas":
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
        payload["image"] = images if isinstance(images, list) else [images]
    if args.get("input_audio"):
        payload["audio"] = args.get("input_audio")
    print(
        f"[alphart-agent] calling internal relay video session_id={_ctx().get('session_id')} "
        f"provider={_log_model_value(payload.get('provider'))} "
        f"model={_log_model_value(payload.get('model'))} url={relay_url}",
        flush=True,
    )
    try:
        resp = requests.post(
            relay_url,
            json=payload,
            headers=_internal_relay_headers(),
            timeout=_backend_tool_timeout(),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Alphart relay request failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] internal relay video response status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
        flush=True,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        return _system_busy_tool_error()
    selected_provider = decoded.get("provider") if isinstance(decoded, dict) else args.get("provider")
    selected_model = decoded.get("model") if isinstance(decoded, dict) else args.get("model")
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
        "task_id": decoded.get("id") if isinstance(decoded, dict) else "",
        "provider": selected_provider,
        "model": selected_model,
    }
    return json.dumps({"status": "success", "result": result}, ensure_ascii=False)


def _handle_alphart_generate_audio(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    requested_duration = int(args.get("duration_seconds") or _ctx().get("duration_seconds") or 0)
    if _ctx().get("app_scope") == "canvas":
        requested_duration = max(5, min(15, requested_duration or 5))
    tool = _pick_tool("audio", args)
    _set_tool_defaults(args, tool)
    selected_provider = str(args.get("provider") or "").strip()
    selected_model = str(args.get("model") or "").strip()
    if not selected_provider or not selected_model:
        return _tool_error("No configured audio generation model is available.", "AUDIO_MODEL_NOT_CONFIGURED")
    args["language_type"] = _normalize_audio_language_type(args.get("language_type")) or _normalize_audio_language_type(_ctx().get("audio_language_type"))
    relay_url = _internal_relay_url("audio/speech")
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
    }
    print(
        f"[alphart-agent] calling internal relay audio session_id={_ctx().get('session_id')} "
        f"provider={_log_model_value(selected_provider)} model={_log_model_value(selected_model)} url={relay_url}",
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
                headers=_internal_relay_headers(),
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


def _game_html_feedback(html: str) -> str:
    value = str(html or "").strip()
    lower = value.lower()
    if not value:
        return "game HTML is empty"
    if not (lower.startswith("<!doctype html") or lower.startswith("<html")):
        return "game HTML must start with <!DOCTYPE html> or <html"
    if "<body" not in lower:
        return "game HTML must include a <body> with visible content"
    if "</body>" not in lower:
        return "game HTML is truncated: missing </body>"
    if "</html>" not in lower:
        return "game HTML is truncated: missing </html>"
    if "<script" not in lower and "onclick=" not in lower and "addeventlistener" not in lower:
        return "game HTML must include inline JavaScript interaction code"
    body_start = lower.find("<body")
    body_end = lower.rfind("</body>")
    body_open_end = lower.find(">", body_start)
    if body_start < 0 or body_end <= body_start or body_open_end < 0:
        return "game HTML must include a valid <body> element"
    body = value[body_start + body_open_end + 1 : body_end].strip()
    visible_body = _strip_invisible_game_html(body)
    if not visible_body.strip():
        return "game HTML body must include visible game DOM content directly in <body>, such as a root stage, HUD, playfield/canvas/SVG, instructions, and controls"
    if not re.search(r"<(main|section|article|div|canvas|svg|button|input|label|h[1-6]|p|span)\b", visible_body, re.I | re.S):
        return "game HTML body must include visible game DOM elements directly in <body>, such as a root stage, HUD, playfield/canvas/SVG, instructions, and controls"
    if re.search(r"position\s*:\s*fixed", value, re.I):
        return "game CSS must not use position:fixed because it can escape the 1920x1080 game stage; use absolute positioning inside the stage instead"
    if re.search(r"(?:width|min-width|max-width)\s*:\s*(?:19[3-9]\d|[2-9]\d{3,})px", value, re.I):
        return "game CSS has an element wider than the 1920px stage; keep every panel/widget inside the 1920x1080 safe area"
    if re.search(r"(?:height|min-height|max-height)\s*:\s*(?:10[9]\d|1[1-9]\d{2}|[2-9]\d{3,})px", value, re.I):
        return "game CSS has an element taller than the 1080px stage; keep every panel/widget inside the 1920x1080 safe area"
    if not re.search(r"\b1920\b", value) or not re.search(r"\b1080\b", value):
        return "game HTML must define an exact 1920x1080 logical stage and scale that whole stage to the viewport"
    if not re.search(r"transform\s*:\s*scale|scale\s*\(", value, re.I):
        return "game HTML must include scale-to-fit logic for the 1920x1080 stage so it shows completely in a new tab and iframe"
    return ""


def _prepare_game_html_for_upload(html: str) -> str:
    value = str(html or "")
    if "alphart-game-fit-stage" in value or "__ALPHART_GAME_FIT__" in value:
        return value
    if (
        re.search(r"\b1920\b", value)
        and re.search(r"\b1080\b", value)
        and re.search(r"transform\s*:\s*scale|scale\s*\(", value, re.I)
    ):
        return value

    style = """<style id="alphart-game-fit-style">
html,body{margin:0!important;width:100%!important;height:100%!important;overflow:hidden!important;}
body{background:#0b1020;display:block!important;}
*,*::before,*::after{box-sizing:border-box;}
#alphart-game-fit-stage{position:fixed;inset:0;display:grid;place-items:center;overflow:hidden;background:inherit;}
#alphart-game-fit-content{position:relative;width:1920px!important;height:1080px!important;transform-origin:top left;overflow:hidden;max-width:none!important;max-height:none!important;will-change:transform;}
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
      content.style.transform="none";
      var logicalW=1920;
      var logicalH=1080;
      content.style.width=logicalW+"px";
      content.style.height=logicalH+"px";
      var scale=Math.min(window.innerWidth/logicalW,window.innerHeight/logicalH);
      if(!isFinite(scale)||scale<=0)scale=1;
      content.style.transform="scale("+scale+")";
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
        "org_no": args.get("org_no") or _ctx().get("storage_prefix"),
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
    html = index_path.read_text(encoding="utf-8")
    feedback = _game_html_feedback(html)
    if feedback:
        raise RuntimeError(feedback)
    prepared_index_html = _prepare_game_html_for_upload(html)

    s3, bucket, index_key = _game_s3_client(target)
    prefix = _game_object_prefix(target)
    uploaded = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        rel = resolved.relative_to(root).as_posix()
        key = f"{prefix}/{_safe_game_rel_path(rel)}"
        body = prepared_index_html.encode("utf-8") if rel == "index.html" else resolved.read_bytes()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=_guess_content_type(rel),
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
    feedback = _game_html_feedback(index_html)
    if feedback:
        raise RuntimeError(feedback)
    prepared_index_html = _prepare_game_html_for_upload(index_html)
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
                feedback = _game_html_feedback(html)
                if feedback:
                    return _tool_error(feedback, code="GAME_VALIDATION_ERROR")
                _upload_game_html(target, html)
            else:
                return _tool_error(f"game artifact path does not exist: {artifact_path}")
        elif has_files:
            _upload_game_files(target, files)
        else:
            feedback = _game_html_feedback(html)
            if feedback:
                return _tool_error(feedback, code="GAME_VALIDATION_ERROR")
            _upload_game_html(target, html)
    except Exception as exc:
        code = "GAME_VALIDATION_ERROR" if str(exc).lower().startswith("game html") or str(exc).lower().startswith("game css") or str(exc).lower().startswith("game javascript") or str(exc).lower().startswith("game must") else ""
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
            "title": {"type": "string"},
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
            "wait": {"type": "boolean", "default": False},
        },
        "required": ["prompt"],
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
    toolset="alphart-edu",
    schema=CANVAS_CREATE_NODE_SCHEMA,
    handler=_handle_canvas_create_node,
    is_async=False,
)
registry.register(
    name="canvas_update_node",
    toolset="alphart-edu",
    schema=CANVAS_UPDATE_NODE_SCHEMA,
    handler=_handle_canvas_update_node,
    is_async=False,
)
registry.register(
    name="canvas_connect_nodes",
    toolset="alphart-edu",
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
    toolset="alphart-edu",
    schema=CANVAS_GENERATE_VIDEO_SCHEMA,
    handler=_handle_alphart_generate_video,
    is_async=False,
)
registry.register(
    name="generate_video",
    toolset="alphart-edu",
    schema={**CANVAS_GENERATE_VIDEO_SCHEMA, "name": "generate_video"},
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
        "terminal, or process; put the complete HTML directly in this tool's html argument, "
        "or pass artifact_dir/artifact_path when a game skill produced a directory containing index.html and assets. "
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
            "artifact_dir": {
                "type": "string",
                "description": (
                    "Optional local directory artifact containing index.html and any asset files. "
                    "When provided, the tool uploads every file recursively under the same public game S3 prefix."
                ),
            },
            "artifact_path": {
                "type": "string",
                "description": (
                    "Optional local artifact path. If it is a directory, it must contain index.html and assets are uploaded recursively. "
                    "If it is a file, it is treated as index.html."
                ),
            },
            "files": {
                "type": "array",
                "description": (
                    "Optional in-memory artifact files. Must include path='index.html'. "
                    "Each file may provide content text or base64 bytes plus optional content_type."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "base64": {"type": "string"},
                        "content_type": {"type": "string"},
                    },
                    "required": ["path"],
                },
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
        "required": ["prompt"],
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
