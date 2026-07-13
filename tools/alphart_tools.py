#!/usr/bin/env python3
"""Alphart app tool bridge.

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


def _tool_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _system_busy_tool_error() -> str:
    return "generate fail"


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
    value = str(
        _ctx().get("backend_url")
        or os.getenv("ALPHART_EDU_BACKEND_URL")
        or os.getenv("CANVAS_BACKEND_URL")
        or ""
    ).strip()
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
        os.getenv("ALPHART_AGENT_TOKEN")
        or os.getenv("CANVAS_AGENT_TOKEN")
        or os.getenv("HERMES_AGENT_TOKEN")
        or ""
    ).strip()


def _call_backend_tool(tool_name: str, args: Dict[str, Any], confirm: bool = False) -> str:
    backend_url = _backend_url()
    if not backend_url:
        return _tool_error("ALPHART_EDU_BACKEND_URL is not configured")
    token = _auth_token()
    service_token = _service_token()

    merged_args = dict(args or {})
    merged_args.setdefault("session_id", _ctx().get("session_id"))
    merged_args.setdefault("user_uuid", _ctx().get("user_uuid"))
    if _ctx().get("storage_prefix"):
        merged_args.setdefault("storage_prefix", _ctx().get("storage_prefix"))
        merged_args.setdefault("org_no", _ctx().get("storage_prefix"))
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
        f"[alphart-agent] calling backend tool name={tool_name} session_id={payload['session_id']} backend_url={backend_url}",
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
            timeout=int(
                os.getenv("ALPHART_EDU_BACKEND_TOOL_TIMEOUT_SECONDS")
                or os.getenv("CANVAS_BACKEND_TOOL_TIMEOUT_SECONDS", "900")
            ),
        )
    except requests.RequestException as exc:
        return _tool_error(f"Canvas backend request failed: {exc}")
    try:
        decoded = resp.json()
    except ValueError:
        decoded = {"raw": resp.text}
    response_preview = (resp.text or "").replace("\n", " ")[:500]
    print(
        f"[alphart-agent] backend tool response name={tool_name} status={resp.status_code} bytes={len(resp.text)} body={response_preview}",
        flush=True,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        return _system_busy_tool_error()
    return json.dumps(decoded, ensure_ascii=False)


def _handle_write_plan(args: Dict[str, Any], **_: Any) -> str:
    return _call_backend_tool("write_plan", args or {}, confirm=False)


def _handle_alphart_generate_image(args: Dict[str, Any], **_: Any) -> str:
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


def _handle_alphart_generate_video(args: Dict[str, Any], **_: Any) -> str:
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
            "Define the learning goal, audience, and winning objective.",
            "Extract a content_facts ledger: exact facts, formulas, definitions, vocabulary, units, correct answers, answer option groups, constraints, and misconceptions that must be preserved.",
            "Choose a playable pixel-game pattern that fits the concept; avoid a static form or plain card quiz unless explicitly requested.",
            "Map learning content into player actions, hazards, collectibles, levels, feedback, and win/fail conditions.",
            "Design the screen layout with a pixel playfield, HUD, safe text panels, and responsive controls.",
            "Generate one complete self-contained HTML file with inline CSS/JS and no external assets.",
            "Review content accuracy, playability, layout overflow, sprite readability, and interaction states before finalizing.",
        ],
    }


def _default_game_layout_requirements() -> Dict[str, Any]:
    return {
        "logical_width": 1920,
        "logical_height": 1080,
        "responsive": "The game must use a fixed 1920x1080 logical stage and may only scale that whole stage visually to fit smaller browser/iframe viewports.",
        "visual_style": "Default to a simple pixel-art game: blocky sprites, tile/grid playfield, crisp edges, limited high-contrast palette, 8-bit inspired UI.",
        "anti_form_rule": "Do not make a plain web form, static worksheet, or button-only quiz unless the user explicitly asks for a quiz/form.",
        "playfield": "Use a real game area with player movement, collectibles/hazards/targets, score/progress, and visible feedback.",
        "stage_rule": "The result page must include a visible root game stage with width:1920px and height:1080px. Center it in the page and scale it with transform: scale(...) from transform-origin: top left when the viewport is smaller.",
        "safe_area": "Keep all text, buttons, sprites, score panels, and dialogs inside the 1920x1080 visible game frame.",
        "overflow_policy": "No clipped text, horizontal scroll, overlapping cards, or elements outside their parent border.",
        "typography": "Use readable font sizes and wrap long labels instead of shrinking below legibility.",
        "controls": "Support keyboard and mouse/touch. On-screen controls must remain reachable on desktop and mobile sizes.",
    }


def _game_html_scaffold_contract() -> str:
    return """Use this coding-agent scaffold shape and fill the TODO sections with the requested game:
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>...</title>
  <style>
    html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#050816}
    body{display:grid;place-items:center}
    #viewport{position:relative;width:100vw;height:100vh;overflow:hidden;display:grid;place-items:center}
    #game-root{position:absolute;width:1920px;height:1080px;overflow:hidden;transform-origin:top left;box-sizing:border-box}
    /* TODO: pixel-art HUD, 1920x1080 playfield, side panel, player, hazards, collectibles, dialogs */
  </style>
</head>
<body>
  <div id="viewport">
    <main id="game-root" role="application" aria-label="Educational game">
      <!-- TODO: visible first-paint HUD, playfield/canvas/SVG, instructions, controls, win/fail UI -->
    </main>
  </div>
  <script>
    const STAGE_W=1920, STAGE_H=1080, root=document.getElementById('game-root');
    function fitStage(){const s=Math.min(innerWidth/STAGE_W,innerHeight/STAGE_H);root.style.transform=`scale(${s})`;root.style.left=`${(innerWidth-STAGE_W*s)/2}px`;root.style.top=`${(innerHeight-STAGE_H*s)/2}px`;}
    addEventListener('resize',fitStage);fitStage();
    addEventListener('message',function(e){if(e.data&&e.data.type==='mute'){document.querySelectorAll('audio,video').forEach(function(el){el.muted=!!e.data.muted;});}});
    // TODO: game state, input, collision/answer checks, precise educational feedback, restart/win/fail flow
  </script>
</body>
</html>"""


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
            "No text is clipped or hidden behind another element.",
            "Buttons, score, progress, dialogs, and game objects have enough spacing.",
            "The layout works at common 16:9, 4:3, tablet, and mobile viewport sizes.",
        ],
        "interaction": [
            "Start/restart flow works.",
            "There is an actual play loop: move/choose/collect/avoid/solve, then receive immediate feedback.",
            "Player can win, fail, or complete a level; the state is visible.",
            "Win/fail/completion state is visible.",
            "Keyboard, mouse, and touch interactions are not blocked.",
        ],
        "playability": [
            "Not just a form or static card.",
            "At least one sprite/player/object moves or changes in response to input.",
            "Challenge has pacing: timer, level, score target, hazards, or progression.",
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
        return "game HTML body must include visible game DOM content"
    if not re.search(r"<(main|section|article|div|canvas|svg|button|input|label|h[1-6]|p|span)\b", visible_body, re.I | re.S):
        return "game HTML body must include visible game DOM content"
    if not re.search(r"\b(game|board|canvas|play|start|score|level|timer|hud|player|sprite|challenge|mission|restart|win|lose|control|controls)\b", visible_body, re.I | re.S):
        return "game HTML body must include visible game DOM content such as a playfield, HUD/score, instructions, or controls"
    if "1920" not in lower or "1080" not in lower:
        return "game result page must use a fixed 1920x1080 logical game window/stage and scale that stage to fit the viewport"
    return ""


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
        f"{backend_url}/api/v1/agent/game-upload-target",
        json=payload,
        headers={
            **({"Authorization": f"Bearer {service_token or token}"} if (service_token or token) else {}),
            **({"X-Hermes-Agent-Token": service_token} if service_token else {}),
        },
        timeout=int(
            os.getenv("ALPHART_EDU_BACKEND_TOOL_TIMEOUT_SECONDS")
            or os.getenv("CANVAS_BACKEND_TOOL_TIMEOUT_SECONDS", "900")
        ),
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"game upload target failed {resp.status_code}: {resp.text[:300]}")
    decoded = resp.json()
    if not isinstance(decoded, dict) or not decoded.get("url"):
        raise RuntimeError("game upload target response is invalid")
    decoded.setdefault("upload_url", decoded.get("url"))
    return decoded


def _upload_game_html(target: Dict[str, Any], html: str) -> None:
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
    s3 = boto3.client("s3", **client_kwargs)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=html.encode("utf-8"),
        ContentType=str(target.get("content_type") or "text/html; charset=utf-8"),
        ACL="public-read",
    )


def _handle_alphart_generate_game(args: Dict[str, Any], **_: Any) -> str:
    args = dict(args or {})
    args.setdefault("game_plan", _default_game_plan())
    args.setdefault("layout_requirements", _default_game_layout_requirements())
    args.setdefault("review_checklist", _default_game_review_checklist())
    args.setdefault("html_scaffold", _game_html_scaffold_contract())
    html = str(args.get("html") or args.get("index_html") or "").strip()
    if not html:
        # Avoid asking the chat model to serialize a full HTML document into a
        # function-call JSON argument. The Go backend can generate, review, and
        # upload the game from the compact prompt safely.
        return _call_backend_tool("canvas_generate_game", args, confirm=False)
    feedback = _game_html_feedback(html)
    if feedback:
        return _tool_error(feedback)
    try:
        target = _request_game_upload_target(args)
        _upload_game_html(target, html)
    except Exception as exc:
        return _tool_error(str(exc))
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
    tool = _pick_tool("audio", args)
    args.setdefault("provider", tool.get("provider"))
    args.setdefault("model", tool.get("model") or tool.get("name") or tool.get("key"))
    tool_name = str(tool.get("id") or "").strip()
    if not tool_name:
        tool_name = f"transcribe_audio_by_{_slug(args.get('provider'))}_{_slug(args.get('model'))}"
    return _call_backend_tool(tool_name, args, confirm=False)


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
    toolset="alphart-edu",
    schema=WRITE_PLAN_SCHEMA,
    handler=_handle_write_plan,
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

CANVAS_GENERATE_GAME_SCHEMA = {
    "name": "canvas_generate_game",
    "description": (
        "Create an educational HTML game through the Alphart Edu Go backend. "
        "Call this with a compact prompt; do not serialize a full HTML document into the tool arguments. "
        "Default style should be a simple pixel-art game with a real play loop, not a plain "
        "web form or static quiz. Good patterns include platformer, top-down maze, arcade "
        "matcher, drag-and-drop sorter, physics launcher, simulation sandbox, boss challenge, "
        "and story quest. Because this is for an education platform, preserve content "
        "precision: formulas, units, definitions, terminology, names, dates, and causal "
        "relationships must stay correct. Use this for 'make a game that teaches/explains ...' or "
        "'create a quiz/game about ...' requests. The backend generates, reviews, uploads, and retries "
        "the complete HTML. If html is omitted, this tool will route prompt-only generation to the backend."
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
                    "Optional complete self-contained playable game HTML only when it already exists. Prefer omitting this field so the Go backend can generate/review/upload the HTML. If provided, it must start with <!DOCTYPE html>, "
                    "include visible first-paint game DOM in <body> (game container/playfield, HUD or score/progress, "
                    "instructions, and controls), inline CSS/JS, closing </body></html>, and a fixed 1920x1080 logical "
                    "game stage that scales as a whole to fit smaller viewports. Do not rely on JavaScript to create "
                    "the only visible game DOM after load."
                ),
            },
            "game_plan": {
                "type": "object",
                "description": (
                    "Required planning payload: learning goal, audience, precise knowledge points, "
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
                    "no overlap, readable labels, and all controls inside the game frame."
                ),
            },
            "review_checklist": {
                "type": "object",
                "description": (
                    "QA checklist used after generation. Include content accuracy, layout/frame "
                    "fit, border overflow, text clipping, interaction states, and mobile/desktop fit."
                ),
            },
            "html_scaffold": {
                "type": "string",
                "description": "Coding-agent scaffold that the generated HTML should follow: fixed 1920x1080 stage, viewport scaler, visible first-paint DOM, and mute listener.",
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
