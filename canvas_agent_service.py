#!/usr/bin/env python3
"""HTTP service wrapper for running Hermes as the Alphart Canvas agent."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from run_agent import AIAgent
from tools.canvas_tools import canvas_context


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


def _provider_config(req: CanvasChatRequest) -> Dict[str, Any]:
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


def _system_prompt(req: CanvasChatRequest) -> str:
    parts = [
        req.system_prompt.strip(),
        "You are the Alphart Canvas AI agent.",
        "Use only Canvas tools for media operations.",
        "Use write_plan for short execution plans.",
        "Use canvas_generate_image for image creation and canvas_generate_video for video task submission.",
        "Do not claim media was generated until the tool returns a backend result.",
    ]
    tool_lines = _selected_tool_lines(req.tool_list)
    if tool_lines:
        parts.append("Selected Canvas tools:\n" + "\n".join(tool_lines))
    return "\n\n".join(part for part in parts if part)


def _public_messages(messages: List[Any]) -> List[Any]:
    out: List[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        cleaned = {k: v for k, v in msg.items() if not str(k).startswith("_")}
        out.append(cleaned)
    return out


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

    messages = [msg for msg in req.messages if isinstance(msg, dict)]
    last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), -1)
    if last_user_index >= 0:
        user_message = _message_text(messages[last_user_index])
        conversation_history = messages[:last_user_index]
    else:
        user_message = _message_text(messages[-1]) if messages else ""
        conversation_history = messages[:-1] if messages else []
    if not user_message:
        raise HTTPException(status_code=400, detail="user message is required")

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
        message = _string(args[0] if args else kwargs.get("message"))
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
    }
    agent = AIAgent(
        base_url=endpoint,
        api_key=api_key,
        provider=provider,
        api_mode=_string(config.get("api_mode")) or "chat_completions",
        model=model,
        enabled_toolsets=["alphart-canvas"],
        max_iterations=int(config.get("max_iterations") or config.get("retry") or 8),
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

    with canvas_context(context):
        result = agent.run_conversation(
            user_message,
            system_message=_system_prompt(req),
            conversation_history=conversation_history,
            task_id=req.session_id or None,
        )

    response_messages = _public_messages(result.get("messages") or [])
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

    agent = AIAgent(
        base_url=endpoint,
        api_key=api_key,
        provider=provider,
        api_mode=_string(config.get("api_mode")) or "chat_completions",
        model=model,
        enabled_toolsets=[],
        max_iterations=1,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )
    result = agent.run_conversation(
        source,
        system_message=(
            "Create a short chat session title. Return only the title, no quotes, "
            "no markdown, no punctuation-only text. Max 8 words."
        ),
        conversation_history=[],
    )
    raw_title = _string(result.get("final_response"))
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
