#!/usr/bin/env python3
"""HTTP service wrapper for running Hermes as the Alphart Canvas agent."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

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


def _input_image_ids_from_text(text: str) -> List[str]:
    return re.findall(r'<image[^>]*\bfile_id="([^"]+)"', text or "")


def _media_intent(text: str) -> str:
    value = (text or "").lower()
    if not value.strip():
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
        "描述",
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
    if not has_creation and not re.search(r"<(?:image_quantity|aspect_ratio|input_images)\b", value):
        return ""
    if any(word in value for word in video_words):
        return "video"
    if any(word in value for word in image_words):
        return "image"
    if any(word in value for word in ("draw", "render", "paint", "sketch", "illustrate", "画", "绘制")):
        return "image"
    return ""


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


def _append_visible_generated_media(messages: List[Any]) -> List[Any]:
    out = list(messages or [])
    for msg in messages or []:
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
                out.append({"role": "assistant", "content": f"Generated video: {url}"})
    return out


def _forced_media_tool_messages(user_message: str, response_messages: List[Any]) -> List[Dict[str, Any]]:
    intent = _media_intent(user_message)
    if not intent or _generation_tool_called(response_messages, intent):
        return []
    if not _selected_media_tools(intent):
        return []

    call_id = str(uuid.uuid4())
    if intent == "image":
        args: Dict[str, Any] = {
            "prompt": user_message,
            "tool_call_id": call_id,
            "image_quantity": _quantity_from_text(user_message),
        }
        aspect_ratio = _aspect_ratio_from_text(user_message)
        if aspect_ratio:
            args["aspect_ratio"] = aspect_ratio
        result = _handle_canvas_generate_image(args)
        tool_name = "canvas_generate_image"
        final_text = (
            "Image generation has been submitted."
            if _tool_result_success(result)
            else "Image generation failed. Please check the tool result."
        )
    else:
        args = {
            "prompt": user_message,
            "tool_call_id": call_id,
        }
        aspect_ratio = _aspect_ratio_from_text(user_message)
        if aspect_ratio:
            args["aspect_ratio"] = aspect_ratio
        result = _handle_canvas_generate_video(args)
        tool_name = "canvas_generate_video"
        final_text = (
            "Video generation has been submitted."
            if _tool_result_success(result)
            else "Video generation failed. Please check the tool result."
        )

    return [
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
- When the user message contains <input_images> XML, extract file_id values and pass them as input_images.
- If more than one input image is present, prefer a selected image tool that supports multiple input_images.
- If the request includes facial expression, mood, emotion, age, gender, region, or cultural constraints, add precise expression-control keywords to the prompt and avoid unsafe or culturally forbidden expression details.

VIDEO CREATION RULES:
- Use video generation tools for video tasks.
- You may generate needed storyboard/keyframe images first, then call video generation using those images, or directly generate video from text if that better fits the request.
- If input images are provided, pass file_id values as input_images.
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
        user_message = _message_text(messages[last_user_index])
        conversation_history = messages[:last_user_index]
    else:
        user_message = _message_text(messages[-1]) if messages else ""
        conversation_history = messages[:-1] if messages else []
    if not user_message:
        raise HTTPException(status_code=400, detail="user message is required")
    input_image_ids = _input_image_ids_from_text(user_message)

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
        "input_image_ids": input_image_ids,
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
        result = agent.run_conversation(
            user_message,
            system_message=_system_prompt(req),
            conversation_history=conversation_history,
            task_id=req.session_id or None,
        )

    response_messages = _public_messages(result.get("messages") or [])
    with canvas_context(context):
        response_messages.extend(_forced_media_tool_messages(user_message, response_messages))
    response_messages = _append_visible_generated_media(response_messages)
    events = [*_events_from_messages(response_messages), *events]
    return {
        "status": "ok",
        "final_response": result.get("final_response") or "",
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
        "failed": bool(result.get("failed")),
    }


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
