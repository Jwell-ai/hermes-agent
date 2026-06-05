#!/usr/bin/env python3
"""HTTP service wrapper for running Hermes as the Alphart Canvas agent."""

from __future__ import annotations

import json
import os
import re
import uuid
import base64
from urllib.parse import quote
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import requests

from run_agent import AIAgent
from tools.canvas_tools import (
    _handle_canvas_generate_image,
    _handle_canvas_generate_video,
    _selected_tools,
    canvas_context,
)


class CanvasChatRequest(BaseModel):
    session_id: str = ""
    canvas_id: str = ""
    user_id: str = ""
    user_uuid: str = ""
    auth_token: str = ""
    messages: List[Any] = Field(default_factory=list)
    text_model: Dict[str, Any] = Field(default_factory=dict)
    tool_list: List[Any] = Field(default_factory=list)
    model_configs: Dict[str, Any] = Field(default_factory=dict)
    backend_url: str = ""
    system_prompt: str = ""


class CanvasTitleRequest(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    text_model: Dict[str, Any] = Field(default_factory=dict)
    model_configs: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Alphart Canvas Hermes Agent", version="1.0.0")
SYSTEM_BUSY_MESSAGE = "System busy, please try again later."


def _service_token() -> str:
    return os.getenv("HERMES_AGENT_TOKEN", "").strip()


def _check_auth(authorization: Optional[str]) -> None:
    token = _service_token()
    if not token:
        return
    expected = f"Bearer {token}"
    if (authorization or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid hermes agent token")


def _string(value: Any) -> str:
    return str(value or "").strip()


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


def _backend_media_url(req: CanvasChatRequest, ref: Dict[str, Any]) -> str:
    raw_url = _string(ref.get("url") or ref.get("uri"))
    if raw_url:
        return raw_url
    object_name = _string(ref.get("s3_object_name") or ref.get("object_name") or ref.get("key"))
    if not object_name:
        return ""
    backend_url = (req.backend_url or os.getenv("CANVAS_BACKEND_URL", "http://localhost:57988")).rstrip("/")
    file_id = _string(ref.get("file_id")) or "media"
    return f"{backend_url}/api/v1/files/{file_id}?s3_object_name={quote(object_name, safe='')}"


def _download_image_as_data_url(req: CanvasChatRequest, ref: Dict[str, Any]) -> str:
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
        print(f"[canvas-agent] image content hydrate failed url={url} error={exc}", flush=True)
        return ""
    mime_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip()
    if not mime_type.startswith("image/"):
        mime_type = _string(ref.get("mime_type") or ref.get("mimeType")) or "image/png"
    if not mime_type.startswith("image/"):
        return ""
    return f"data:{mime_type};base64,{base64.b64encode(resp.content).decode('ascii')}"


def _prepare_chat_content_for_model(req: CanvasChatRequest, content: Any) -> Any:
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
    provider = _string(req.text_model.get("provider"))
    model = _string(req.text_model.get("model"))
    def with_model_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(raw)
        models = raw.get("models")
        if isinstance(models, dict) and model and isinstance(models.get(model), dict):
            merged.update(models[model])
        return merged
    config = req.model_configs.get("text")
    if isinstance(config, dict) and provider:
        raw = config.get(provider)
        if isinstance(raw, dict):
            return with_model_config(raw)
    if isinstance(req.model_configs, dict) and provider:
        raw = req.model_configs.get(provider)
        if isinstance(raw, dict):
            return with_model_config(raw)
    return {}


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
            images.append(
                {
                    "s3_object_name": object_name,
                    "file_id": _string(ref.get("file_id")),
                    "width": _string(ref.get("width")),
                    "height": _string(ref.get("height")),
                }
            )
            continue
        file_id = _string(ref.get("file_id"))
        if file_id:
            images.append(file_id)
    return images


def _asset_input_image(asset: Dict[str, Any]) -> Any:
    object_name = _string(asset.get("s3_object_name") or asset.get("object_name") or asset.get("key"))
    if object_name:
        return {
            "s3_object_name": object_name,
            "file_id": _string(asset.get("file_id") or asset.get("id")),
            "width": _string(asset.get("width")),
            "height": _string(asset.get("height")),
        }
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
    has_creation = any(word in value for word in creation_words)
    has_regeneration = _regeneration_intent(value)
    if not has_creation and not (has_regeneration and (has_image_context or has_video_context)):
        return ""
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
    expected = "image" if media_type == "image" else "video"
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
    expected = "image" if media_type == "image" else "video"
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
                if not expected and ("generate_image" in name or "canvas_generate_image" in name or "generate_video" in name or "canvas_generate_video" in name):
                    return True
        if msg.get("role") == "tool":
            name = _string(msg.get("name")).lower()
            if expected == "image" and "image" in name:
                return True
            if expected == "video" and "video" in name:
                return True
            if not expected and ("image" in name or "video" in name):
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
        if expected == "" and "image" not in name and "video" not in name:
            continue
        if not _tool_result_success(msg.get("content")):
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
    try:
        decoded = json.loads(result)
    except (TypeError, ValueError):
        return False
    if isinstance(decoded, dict) and decoded.get("success") is False:
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
        for key in ("assets", "images", "videos", "outputs"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    else:
        candidates = []

    assets: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        url = _string(item.get("url") or item.get("image_url") or item.get("video_url"))
        mime_type = _string(item.get("mime_type") or item.get("mimeType"))
        if not url:
            continue
        if media_type == "image" and mime_type and not mime_type.startswith("image/"):
            continue
        if media_type == "video" and mime_type and not mime_type.startswith("video/"):
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
    r"https?://[^\s)'\"<>]+(?:\.(?:png|jpe?g|webp|gif|mp4|m4v|mov|f4v|flv|webm|ogg))(?:\?[^\s)'\"<>]*)?",
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
    cleaned = re.sub(r"^\s*(Generated image|Generated video|Image|Video|图片|图像|视频|影片|生成图片|生成视频)\s*[:：]?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
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


def _forced_media_tool_messages(
    user_message: str,
    response_messages: List[Any],
    scan_messages: Optional[List[Any]] = None,
    has_image_context: bool = False,
    has_video_context: bool = False,
    input_images: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    if _media_analysis_intent(user_message.lower()):
        return []
    intent = _media_intent(user_message, has_image_context=has_image_context, has_video_context=has_video_context)
    if not intent:
        return []
    current_messages = scan_messages if scan_messages is not None else response_messages
    if _generation_tool_completed(current_messages, intent):
        return []

    call_id = str(uuid.uuid4())
    if intent == "image":
        args: Dict[str, Any] = {
            "prompt": user_message,
            "tool_call_id": call_id,
            "image_quantity": _quantity_from_text(user_message),
        }
        if has_image_context and input_images:
            args["input_images"] = input_images
        aspect_ratio = _aspect_ratio_from_text(user_message)
        if aspect_ratio:
            args["aspect_ratio"] = aspect_ratio
        print(
            f"[canvas-agent] forcing image generation session_intent={intent} tool_count={len(_selected_media_tools(intent))}",
            flush=True,
        )
        result = _handle_canvas_generate_image(args)
        tool_name = "canvas_generate_image"
        final_text = (
            "Image generation has been submitted."
            if _tool_result_success(result)
            else "generate fail"
        )
    else:
        args = {
            "prompt": user_message,
            "tool_call_id": call_id,
        }
        aspect_ratio = _aspect_ratio_from_text(user_message)
        if aspect_ratio:
            args["aspect_ratio"] = aspect_ratio
        print(
            f"[canvas-agent] forcing video generation session_intent={intent} tool_count={len(_selected_media_tools(intent))}",
            flush=True,
        )
        result = _handle_canvas_generate_video(args)
        tool_name = "canvas_generate_video"
        final_text = (
            "Video generation has been submitted."
            if _tool_result_success(result)
            else "generate fail"
        )

    plan_text = (
        "Plan:\n"
        "1. Use the user's request and referenced media as generation context.\n"
        "2. Create a new generation request instead of reusing an old result.\n"
        "3. Return the newly generated media result."
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


def _canvas_agent_prompt(req: CanvasChatRequest) -> str:
    tool_lines = _selected_tool_lines(req.tool_list)
    selected_tools = "\n".join(tool_lines) if tool_lines else "- No image/video model selected. Ask the user to select a model before generation."
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
- If the user asks to explain, describe, analyze, summarize, caption, identify, or understand an attached image/video, answer with the text/chat model. Do not call image/video generation tools.
- For obvious image/video generation or editing tasks, a generation tool call is mandatory.
- For simple media requests, call canvas_generate_image/canvas_generate_video directly. Do not stop after a plan.
- For complex media requests, you may call write_plan first, but you must continue to the generation tool after the plan result.
- Do not ask for approval before media generation unless the backend returns a confirmation request.
- Do not call multiple tools in the same assistant turn. Always wait for one tool result before making another tool call.
- If a tool call fails, explain the error to the user and do not retry automatically.
- Pay attention to requested quantity. If the user asks for 20 images, keep exactly 20 in the plan and generation batches. If no quantity is specified, assume 1.

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

ERROR HANDLING:
- Read backend tool errors carefully.
- Never retry the same failing tool call automatically.
- Never call the same tool with the same parameters again without user confirmation.
- Explain the specific failure and suggest a safer alternative prompt or model.
""".strip()


def _system_prompt(req: CanvasChatRequest) -> str:
    return _canvas_agent_prompt(req)


def _public_messages(messages: List[Any]) -> List[Any]:
    out: List[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        cleaned = {k: v for k, v in msg.items() if not str(k).startswith("_")}
        if "content" in cleaned:
            cleaned["content"] = _message_text(cleaned)
        out.append(cleaned)
    return out


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


def _callback_backend_url(req: CanvasChatRequest) -> str:
    return _string(req.backend_url or os.getenv("CANVAS_BACKEND_URL")).rstrip("/")


def _callback_service_token() -> str:
    return _string(os.getenv("CANVAS_AGENT_TOKEN") or os.getenv("HERMES_AGENT_TOKEN"))


def _post_chat_result_callback(req: CanvasChatRequest, response: Dict[str, Any]) -> None:
    backend_url = _callback_backend_url(req)
    if not backend_url:
        print(
            f"[canvas-agent] chat result callback skipped session_id={req.session_id} reason=missing_backend_url",
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
            f"{backend_url}/api/v1/agent/chat-results",
            json=payload,
            headers={
                **({"Authorization": f"Bearer {token}"} if token else {}),
                **({"X-Hermes-Agent-Token": token} if token else {}),
            },
            timeout=int(os.getenv("CANVAS_BACKEND_CALLBACK_TIMEOUT_SECONDS", "30")),
        )
    except requests.RequestException as exc:
        print(
            f"[canvas-agent] chat result callback failed session_id={req.session_id} error={exc}",
            flush=True,
        )
        return
    preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[canvas-agent] chat result callback response session_id={req.session_id} status={resp.status_code} bytes={len(resp.text)} body={preview}",
        flush=True,
    )


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name) or 0)
    return int(getattr(usage, name, 0) or 0)


def _generate_title_direct(endpoint: str, api_key: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    client = OpenAI(api_key=api_key, base_url=endpoint, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a short chat session title. Return only the title, no quotes, "
                    "no markdown, no punctuation-only text. Max 8 words."
                ),
            },
            {"role": "user", "content": source},
        ],
        max_tokens=32,
        temperature=0.2,
    )
    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return {
        "title": content,
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
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
                events.append({"type": "tool_call", "id": tool_call_id, "name": tool_name})
                args = _tool_call_arguments(tool_call)
                if args and args != "{}":
                    events.append({"type": "tool_call_arguments", "id": tool_call_id, "text": args})
        elif msg.get("role") == "tool":
            tool_call_id = _string(msg.get("tool_call_id"))
            if tool_call_id:
                events.append({"type": "tool_call_result", "id": tool_call_id, "message": msg})
    return events


@app.get("/health")
@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/v1/agent/chats")
def chat(req: CanvasChatRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_auth(authorization)

    provider = _string(req.text_model.get("provider"))
    model = _string(req.text_model.get("model"))
    config = _provider_config(req)
    endpoint = _endpoint(config)
    api_key = _api_key(config)
    if not provider or not model:
        raise HTTPException(status_code=400, detail="text model provider/model is required")
    if not endpoint or not api_key:
        raise HTTPException(status_code=400, detail="text model endpoint/api_key is not configured")

    raw_messages = [msg for msg in req.messages if isinstance(msg, dict)]
    messages = _fix_chat_history(_filter_image_content(raw_messages, req.text_model))
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
    input_images = _input_images_from_text(user_message)
    latest_generated_image = _latest_generated_image_ref(conversation_history)
    if not input_images and latest_generated_image and _media_intent(user_message, has_image_context=True) == "image":
        input_images = [latest_generated_image]
    model_user_message: Any = user_message
    if _model_supports_vision(req.text_model) and _has_media_content(user_content):
        prepared_content = _prepare_chat_content_for_model(req, user_content)
        if isinstance(prepared_content, list) and prepared_content:
            model_user_message = prepared_content

    events: List[Dict[str, Any]] = []

    def on_delta(*args: Any, **kwargs: Any) -> None:
        text = ""
        if args:
            text = _string(args[0])
        if not text:
            text = _string(kwargs.get("delta") or kwargs.get("text"))
        if text:
            events.append({"type": "delta", "text": text})

    def on_status(*args: Any, **kwargs: Any) -> None:
        message = _string(args[1] if len(args) > 1 else (args[0] if args else kwargs.get("message")))
        if message:
            events.append({"type": "status", "message": message})

    context = {
        "session_id": req.session_id,
        "canvas_id": req.canvas_id,
        "user_id": req.user_id,
        "user_uuid": req.user_uuid,
        "auth_token": req.auth_token,
        "backend_url": req.backend_url or os.getenv("CANVAS_BACKEND_URL", "http://localhost:57988"),
        "tool_list": req.tool_list,
        "input_images": input_images,
    }
    with canvas_context(context):
        agent = AIAgent(
            base_url=endpoint,
            api_key=api_key,
            provider=provider,
            api_mode=_string(config.get("api_mode")) or "chat_completions",
            model=model,
            enabled_toolsets=["alphart-canvas"],
            max_iterations=_agent_max_iterations(config),
            quiet_mode=True,
            session_id=req.session_id or None,
            stream_delta_callback=on_delta,
            status_callback=on_status,
            platform="alphart-canvas",
            user_id=req.user_id or req.user_uuid or None,
            chat_id=req.session_id or None,
            skip_memory=True,
            skip_context_files=True,
        )
        try:
            result = agent.run_conversation(
                model_user_message,
                system_message=_system_prompt(req),
                conversation_history=conversation_history,
                task_id=req.session_id or None,
                persist_user_message=user_message,
            )
        except Exception as exc:
            print(
                f"[canvas-agent] chat model failed session_id={req.session_id} provider={provider} model={model} error={exc}",
                flush=True,
            )
            result = {
                "messages": [{"role": "assistant", "content": SYSTEM_BUSY_MESSAGE}],
                "final_response": SYSTEM_BUSY_MESSAGE,
                "model": model,
                "provider": provider,
                "failed": True,
                "error": SYSTEM_BUSY_MESSAGE,
            }

    raw_result_messages = result.get("messages") or []
    response_messages = _public_messages(raw_result_messages)
    response_messages = _current_turn_response_messages(response_messages, conversation_history)
    current_turn_messages = _messages_after_latest_user(response_messages)
    current_media_attempted = _generation_tool_attempted(current_turn_messages)
    current_media_failed = _generation_tool_failed(current_turn_messages)
    reference_image_generation = (
        bool(input_images)
        and not current_media_attempted
        and _media_intent(user_message, has_image_context=True) == "image"
    )
    if reference_image_generation:
        if _generation_tool_completed(current_turn_messages, "image"):
            response_messages = current_turn_messages
        else:
            if response_messages:
                print(
                    f"[canvas-agent] discarding non-tool reference image response session_id={req.session_id} message_count={len(response_messages)}",
                    flush=True,
                )
            response_messages = []
    final_response = _string(result.get("final_response"))
    if current_media_failed:
        final_response = SYSTEM_BUSY_MESSAGE
    if reference_image_generation and not _generation_tool_completed(response_messages, "image"):
        final_response = ""
    if not final_response:
        if reference_image_generation:
            final_response = _last_assistant_text(response_messages)
        else:
            final_response = _last_assistant_text(response_messages) or _last_assistant_text(raw_result_messages)
    with canvas_context(context):
        current_turn_messages = _messages_after_latest_user(response_messages)
        forced_messages = []
        if not current_media_attempted:
            forced_messages = _forced_media_tool_messages(
                user_message,
                response_messages,
                current_turn_messages,
                has_image_context=bool(input_images),
                input_images=input_images,
            )
        response_messages.extend(forced_messages)
    current_turn_messages = _messages_after_latest_user(response_messages)
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
        "[canvas-agent] chat result "
        f"session_id={req.session_id} raw_messages={raw_count} "
        f"public_messages={len(response_messages)} final_response_len={len(final_response)} "
        f"failed={bool(result.get('failed') or empty_result_error)} "
        f"error={empty_result_error}",
        flush=True,
    )
    response = {
        "status": "ok",
        "final_response": final_response,
        "messages": response_messages,
        "events": events,
        "model": result.get("model") or model,
        "provider": result.get("provider") or provider,
        "prompt_tokens": result.get("prompt_tokens") or 0,
        "completion_tokens": result.get("completion_tokens") or 0,
        "total_tokens": result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "interrupted": bool(result.get("interrupted")),
        "failed": bool(result.get("failed") or empty_result_error),
        "error": empty_result_error or _string(result.get("error")),
    }
    _post_chat_result_callback(req, response)
    return response


@app.post("/api/v1/agent/titles")
def title(req: CanvasTitleRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _check_auth(authorization)

    provider = _string(req.text_model.get("provider"))
    model = _string(req.text_model.get("model"))
    config = _provider_config(req)
    endpoint = _endpoint(config)
    api_key = _api_key(config)
    if not provider or not model:
        raise HTTPException(status_code=400, detail="text model provider/model is required")
    if not endpoint or not api_key:
        raise HTTPException(status_code=400, detail="text model endpoint/api_key is not configured")

    source = "\n".join(_message_text(msg) for msg in req.messages if isinstance(msg, dict)).strip()
    if not source:
        raise HTTPException(status_code=400, detail="title source is required")

    result = _generate_title_direct(endpoint, api_key, model, source, config)
    raw_title = _string(result.get("title"))
    cleaned = raw_title.strip().strip("\"'`").splitlines()[0].strip() if raw_title else ""
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip()
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
    uvicorn.run("canvas_agent_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
