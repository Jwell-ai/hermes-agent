#!/usr/bin/env python3
"""Alphart Canvas tool bridge.

The Hermes agent owns the reasoning loop. Canvas still owns auth, credit
billing, S3 persistence, and Seedance polling, so these tools call the Go
backend's normal `/api/v1/tools/execute` endpoint.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, Iterator

import requests

from tools.registry import registry


_canvas_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "canvas_context", default={}
)


@contextlib.contextmanager
def canvas_context(values: Dict[str, Any]) -> Iterator[None]:
    token = _canvas_context.set(dict(values or {}))
    try:
        yield
    finally:
        _canvas_context.reset(token)


def _ctx() -> Dict[str, Any]:
    return _canvas_context.get() or {}


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _system_busy_tool_error() -> str:
    return json.dumps({"success": False, "error": "System busy, please try again later."}, ensure_ascii=False)


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


def _backend_url() -> str:
    value = str(_ctx().get("backend_url") or os.getenv("CANVAS_BACKEND_URL") or "").strip()
    return value.rstrip("/")


def _auth_token() -> str:
    return str(_ctx().get("auth_token") or os.getenv("CANVAS_AUTH_TOKEN") or "").strip()


def _service_token() -> str:
    return str(os.getenv("CANVAS_AGENT_TOKEN") or os.getenv("HERMES_AGENT_TOKEN") or "").strip()


def _call_backend_tool(tool_name: str, args: Dict[str, Any], confirm: bool = False) -> str:
    backend_url = _backend_url()
    if not backend_url:
        return _tool_error("CANVAS_BACKEND_URL is not configured")
    token = _auth_token()
    service_token = _service_token()

    merged_args = dict(args or {})
    merged_args.setdefault("session_id", _ctx().get("session_id"))
    merged_args.setdefault("user_uuid", _ctx().get("user_uuid"))
    if _ctx().get("canvas_id"):
        merged_args.setdefault("canvas_id", _ctx().get("canvas_id"))

    payload = {
        "tool_call_id": str(merged_args.get("tool_call_id") or uuid.uuid4()),
        "session_id": str(_ctx().get("session_id") or merged_args.get("session_id") or ""),
        "tool_name": tool_name,
        "arguments": merged_args,
        "confirm": bool(confirm),
    }
    print(
        f"[canvas-agent] calling backend tool name={tool_name} session_id={payload['session_id']} backend_url={backend_url}",
        flush=True,
    )
    try:
        resp = requests.post(
            f"{backend_url}/api/v1/tools/execute",
            json=payload,
            headers={
                **({"Authorization": f"Bearer {token}"} if token else {}),
                **({"X-Hermes-Agent-Token": service_token} if service_token else {}),
            },
            timeout=int(os.getenv("CANVAS_BACKEND_TOOL_TIMEOUT_SECONDS", "900")),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Canvas backend request failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[canvas-agent] backend tool response name={tool_name} status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
        flush=True,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        return _system_busy_tool_error()
    return json.dumps(decoded, ensure_ascii=False)


def _handle_write_plan(args: Dict[str, Any], **_: Any) -> str:
    return _call_backend_tool("write_plan", args or {}, confirm=False)


def _handle_canvas_generate_image(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if args.get("image_quantity") and not args.get("quantity"):
        args["quantity"] = args.get("image_quantity")
    if not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")
    tool = _pick_tool("image", args)
    args.setdefault("provider", tool.get("provider"))
    args.setdefault("model", tool.get("model") or tool.get("name") or tool.get("key"))
    tool_name = str(tool.get("id") or "").strip()
    if not tool_name:
        tool_name = f"generate_image_by_{_slug(args.get('provider'))}_{_slug(args.get('model'))}"
    return _call_backend_tool(tool_name, args, confirm=bool(tool.get("requires_confirmation")))


def _handle_canvas_generate_video(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    if args.get("duration_seconds") and not args.get("duration"):
        args["duration"] = args.get("duration_seconds")
    if args.get("image_url") and not args.get("input_images"):
        args["input_images"] = [args.get("image_url")]
    if not args.get("input_images") and _ctx().get("input_images"):
        args["input_images"] = _ctx().get("input_images")
    tool = _pick_tool("video", args)
    args.setdefault("provider", tool.get("provider"))
    args.setdefault("model", tool.get("model") or tool.get("name") or tool.get("key"))
    args.setdefault("wait", False)
    tool_name = str(tool.get("id") or "").strip()
    if not tool_name:
        tool_name = f"generate_video_by_{_slug(args.get('provider'))}_{_slug(args.get('model'))}"
    return _call_backend_tool(tool_name, args, confirm=bool(tool.get("requires_confirmation")))


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
                    },
                },
                "description": "Reference images extracted from <input_images> XML. Prefer s3_object_name objects; file_id strings are fallback only.",
            },
            "duration_seconds": {"type": "integer", "description": "Requested video duration in seconds."},
            "resolution": {"type": "string", "description": "Video resolution, for example 480p, 720p, 1080p."},
            "aspect_ratio": {"type": "string", "description": "Video aspect ratio, for example 16:9 or 9:16."},
            "wait": {"type": "boolean", "default": False},
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="write_plan",
    toolset="alphart-canvas",
    schema=WRITE_PLAN_SCHEMA,
    handler=_handle_write_plan,
    is_async=False,
)
registry.register(
    name="canvas_generate_image",
    toolset="alphart-canvas",
    schema=CANVAS_GENERATE_IMAGE_SCHEMA,
    handler=_handle_canvas_generate_image,
    is_async=False,
)
registry.register(
    name="generate_image",
    toolset="alphart-canvas",
    schema={**CANVAS_GENERATE_IMAGE_SCHEMA, "name": "generate_image"},
    handler=_handle_canvas_generate_image,
    is_async=False,
)
registry.register(
    name="canvas_generate_video",
    toolset="alphart-canvas",
    schema=CANVAS_GENERATE_VIDEO_SCHEMA,
    handler=_handle_canvas_generate_video,
    is_async=False,
)
registry.register(
    name="generate_video",
    toolset="alphart-canvas",
    schema={**CANVAS_GENERATE_VIDEO_SCHEMA, "name": "generate_video"},
    handler=_handle_canvas_generate_video,
    is_async=False,
)
