#!/usr/bin/env python3
"""HTTP service wrapper for running Hermes as the Alphart Canvas agent."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
import base64
from urllib.parse import quote
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("canvas_agent")

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI, APIConnectionError as OpenAIConnectionError, APIStatusError as OpenAIStatusError, APITimeoutError as OpenAITimeoutError
from pydantic import BaseModel, Field
import requests

from run_agent import AIAgent
from tools.canvas_tools import (
    _handle_canvas_generate_image,
    _handle_canvas_generate_video,
    _handle_canvas_transcribe_audio,
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
    text_models: List[Dict[str, Any]] = Field(default_factory=list)
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


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)


def _strip_think_tags(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


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
    return _provider_config_for(req.model_configs, req.text_model)


def _provider_config_for(model_configs: Any, text_model: Dict[str, Any]) -> Dict[str, Any]:
    provider = _string(text_model.get("provider"))
    model = _string(text_model.get("model"))
    def with_model_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(raw)
        models = raw.get("models")
        if isinstance(models, dict) and model and isinstance(models.get(model), dict):
            merged.update(models[model])
        return merged
    if not isinstance(model_configs, dict):
        return {}
    config = model_configs.get("text")
    if isinstance(config, dict) and provider:
        raw = config.get(provider)
        if isinstance(raw, dict):
            return with_model_config(raw)
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


def _game_tool_failed(messages: List[Any]) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if "generate_game" not in _string(msg.get("name")).lower():
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
        f"[canvas-agent] audio pipeline: transcribing audio_url={audio_url[:80]}",
        flush=True,
    )
    transcribe_result = _handle_canvas_transcribe_audio(transcribe_args)
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
        f"[canvas-agent] audio pipeline: intent={intent} transcribed_len={len(transcribed_text)}",
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
        gen_result = _handle_canvas_generate_image(gen_args)
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
        gen_result = _handle_canvas_generate_video(gen_args)
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
            f"[canvas-agent] forcing image generation session_intent={intent} quantity={quantity} "
            f"tool_count={len(_selected_media_tools(intent))}",
            flush=True,
        )
        messages: List[Dict[str, Any]] = [{"role": "assistant", "content": plan_text}]
        success_count = 0
        for task_index in range(1, quantity + 1):
            call_id = str(uuid.uuid4())
            task_prompt = user_message
            if quantity > 1:
                task_prompt = f"{user_message} (variation {task_index} of {quantity}: vary composition, angle, lighting, or style)"
            args: Dict[str, Any] = {
                "prompt": task_prompt,
                "tool_call_id": call_id,
                "image_quantity": 1,
            }
            if has_image_context and input_images:
                args["input_images"] = input_images
            if aspect_ratio:
                args["aspect_ratio"] = aspect_ratio
            result = _handle_canvas_generate_image(args)
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

    call_id = str(uuid.uuid4())
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
    selected_tools = "\n".join(tool_lines) if tool_lines else "- No image/video/audio model selected. Ask the user to select a model before generation."
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

AUDIO INPUT RULES:
- Audio is input-only. There is no text-to-speech output.
- When the user message contains an audio_url content part, call canvas_transcribe_audio immediately to get the text.
- After transcription, read the transcribed text, detect the user's intent, and act on it exactly as if the user had typed that text.
- If the transcribed text is an image or video generation command, follow the plan: 1) transcribe, 2) refine prompt, 3) call the generation API.
- If transcription fails, report the failure clearly and stop.

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


_TITLE_SYSTEM = (
    "Create a short chat session title. Return only the title, no quotes, "
    "no markdown, no punctuation-only text. Max 8 words."
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


def _generate_title_anthropic(endpoint: str, api_key: str, model: str, source: str, config: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(config.get("timeout") or config.get("timeout_seconds") or 60)
    base = (endpoint or "https://api.anthropic.com/v1").rstrip("/")
    if not base.endswith("/messages"):
        base += "/messages"
    body = {
        "model": model,
        "max_tokens": 32,
        "system": _TITLE_SYSTEM,
        "messages": [{"role": "user", "content": source}],
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
        "contents": [{"role": "user", "parts": [{"text": source}]}],
        "systemInstruction": {"parts": [{"text": _TITLE_SYSTEM}]},
        "generationConfig": {"maxOutputTokens": 32, "temperature": 0.2},
    }
    headers = {"content-type": "application/json", "authorization": "Bearer " + (api_key or "")}
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
                {"role": "user", "content": source},
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
    input_images = _input_images_from_text(user_message)
    latest_generated_image = _latest_generated_image_ref(conversation_history)
    if not input_images and latest_generated_image and _media_intent(user_message, has_image_context=True) == "image":
        input_images = [latest_generated_image]
    model_user_message: Any = user_message
    if _model_supports_vision(primary_text_model) and _has_media_content(user_content):
        prepared_content = _prepare_chat_content_for_model(req, user_content)
        if isinstance(prepared_content, list) and prepared_content:
            model_user_message = prepared_content

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
        if not provider or not model or not endpoint or not api_key:
            print(
                f"[canvas-agent] skipping unconfigured text model session_id={req.session_id} provider={provider} model={model}",
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
                    attempt_events.append({"type": "delta", "text": text})

            def on_status(*args: Any, **kwargs: Any) -> None:
                message = _string(args[1] if len(args) > 1 else (args[0] if args else kwargs.get("message")))
                if message:
                    attempt_events.append({"type": "status", "message": message})

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
                        f"[canvas-agent] chat model failed session_id={req.session_id} provider={provider} "
                        f"model={model} attempt={attempt}/{retry_count} error={exc}",
                        flush=True,
                    )
                    result = {"failed": True, "error": str(exc), "model": model, "provider": provider}

            if not result.get("failed"):
                events = attempt_events
                break
            print(
                f"[canvas-agent] chat model attempt failed session_id={req.session_id} provider={provider} "
                f"model={model} attempt={attempt}/{retry_count} error={result.get('error')}",
                flush=True,
            )
            events = attempt_events
        if not result.get("failed"):
            break

    if result.get("failed"):
        result = {
            "messages": [{"role": "assistant", "content": SYSTEM_BUSY_MESSAGE}],
            "final_response": SYSTEM_BUSY_MESSAGE,
            "model": result.get("model") or last_model,
            "provider": result.get("provider") or last_provider,
            "failed": True,
            "error": SYSTEM_BUSY_MESSAGE,
        }

    raw_result_messages = result.get("messages") or []
    response_messages = _public_messages(raw_result_messages)
    response_messages = _current_turn_response_messages(response_messages, conversation_history)
    current_turn_messages = _messages_after_latest_user(response_messages)
    current_media_attempted = _generation_tool_attempted(current_turn_messages)
    current_media_failed = _generation_tool_failed(current_turn_messages)
    current_game_failed = _game_tool_failed(current_turn_messages)
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
    if current_media_failed or current_game_failed:
        final_response = SYSTEM_BUSY_MESSAGE
    if reference_image_generation and not _generation_tool_completed(response_messages, "image"):
        final_response = ""
    if not final_response:
        if reference_image_generation:
            final_response = _last_assistant_text(response_messages)
        else:
            final_response = _last_assistant_text(response_messages) or _last_assistant_text(raw_result_messages)
    user_audio_urls = _audio_urls_from_content(user_content)
    with canvas_context(context):
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
        if not forced_messages and not current_media_attempted:
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
        "model": result.get("model") or last_model,
        "provider": result.get("provider") or last_provider,
        "prompt_tokens": result.get("prompt_tokens") or 0,
        "completion_tokens": result.get("completion_tokens") or 0,
        "total_tokens": result.get("total_tokens") or 0,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "interrupted": bool(result.get("interrupted") or current_game_failed),
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
    config_providers = list(req.model_configs.keys()) if isinstance(req.model_configs, dict) else []
    logger.info(
        "title request provider=%r model=%r endpoint=%r has_key=%s config_providers=%s",
        provider, model, endpoint, bool(api_key), config_providers,
    )
    if not provider or not model:
        logger.warning("title 400: provider=%r model=%r", provider, model)
        raise HTTPException(status_code=400, detail="text model provider/model is required")
    if not endpoint or not api_key:
        logger.warning("title 400: endpoint=%r has_key=%s provider=%r config_keys=%s", endpoint, bool(api_key), provider, config_providers)
        raise HTTPException(status_code=400, detail="text model endpoint/api_key is not configured")

    source = "\n".join(_message_text(msg) for msg in req.messages if isinstance(msg, dict)).strip()
    if not source:
        raise HTTPException(status_code=400, detail="title source is required")

    result = _generate_title_direct(provider, endpoint, api_key, model, source, config)
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
