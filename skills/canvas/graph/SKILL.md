# Canvas Graph Orchestration

This skill is loaded only for `app_scope=canvas`. It is the authoritative workflow for interpreting natural-language Canvas requests and operating Canvas nodes and connections.

## Context

- `canvas_id` is the current document. The backend has already enforced organization membership and write permission.
- `canvas_item_id`, when present, is the explicit execution target and is the default execution target.
- `selected_canvas_item_id`, when present without `canvas_item_id`, is only the UI selection. Use its graph entry as context for a new downstream design; do not mutate it unless the user explicitly asks for an edit.
- The backend may provide a graph inventory containing node ids, types, titles, content, media references, and existing lines. Node ids are tool-call-only values.
- A graph `media_ref` is an opaque Canvas object reference. Copy it only into the matching tool's `s3_object_name` reference field when that node is an explicitly selected input; never turn it into a URL or mention it in the user-facing response.
- User mentions such as `@Prompt`, `@Image node`, `@a.mp3`, or equivalent native-language references identify graph inputs. Match by title and graph context; never expose ids in the response.
- Every newly created node gets a concise, human-readable title summarizing its role or content in 2-6 words. Preserve an explicit title from the user; avoid generic titles such as `Image node`, `Video node`, or `Prompt` when a meaningful summary is available.
- The Canvas backend owns persistence, S3 storage, provider relay, task polling, billing, and authorization. Do not use local files, local storage, direct provider credentials, or external storage.

## Intent And Target Selection

First understand the request in its original language. Do not classify intent from a fixed keyword list and do not ask the frontend to classify it.

- If `canvas_item_id` is present, operate only on that existing node. A request to generate or refine affects that node. Do not create a replacement node or silently redirect the result.
- If the user explicitly asks for a new node, a downstream result, or a new design and no selected target is supplied, create the requested graph through the Canvas tools.
- If a selected node is a text/note node and the user asks for media creation, treat the selected text and any named references as inputs for a new downstream graph rather than overwriting the text node.
- If the request is an edit, rename, move, resize, delete, or connection operation, perform only that operation.
- Never reveal internal node ids, organization ids, user ids, provider ids, object keys, credentials, or raw signed URLs.

## Requirement And Capacity Routing

Treat every request as a small production brief before choosing a tool. Extract only what the user actually asked for:

- outcome: text, image, video, audio, or graph operation;
- target: explicit `canvas_item_id`, selected node, named `@` references, or a new downstream result;
- media roles: prompt/script, first/intermediate/last keyframe, soundtrack/background music, or voice print;
- constraints: duration, aspect ratio, resolution, image quality, language, captions, and audio behavior;
- workflow: ordinary generation, an explicitly requested specialist workflow, or analysis of an existing result.

Use only the configured Canvas tools and the model options supplied in the request. Never invent a provider, model, endpoint, duration mode, or unsupported capability. If a requested option is unavailable, keep the graph unchanged and explain the smallest necessary limitation.

Choose the narrowest applicable workflow:

1. `canvas-video-shot-breakdown` is analysis only. Use it only when the user explicitly asks for shot breakdown/analysis and the selected or explicit target is an existing completed video. Call `video_analyze` exactly once; never generate a new video in this workflow.
2. `canvas-video-shotcraft` is for an explicit shotcraft/shot-recipe/shot-card request. Use it to design a new video graph or update the explicit video target, then call exactly one `canvas_generate_video` when generation is requested. If the skill is unavailable, preserve the brief and use the Seedance director workflow instead of failing a normal video request.
3. `canvas-seedance2-video-director` is the default video workflow. Use it for all other video generation, including a new graph with text/image/audio references. Preserve reference roles and call exactly one `canvas_generate_video` for the output node.
4. Image, audio, and text requests use the matching Canvas tool and do not load a video workflow.

Capacity rules are explicit: video duration is supported from 5 to 15 seconds and must be clamped to that range; do not silently change a requested ratio, resolution, quality, language, or model to an invented value. A connected soundtrack/background-music node disables provider-generated audio; a voice-print reference is not a soundtrack. Caption/script text is passed as the production brief, but it is not an audio reference. Audio and caption artifacts are separate Canvas nodes when the user asks for them.

For a new graph, create only the nodes needed for the selected workflow and connect them before generation. For an existing target, update only that target unless the user explicitly asks for a downstream result. Do not duplicate a task because a tool response is delayed or because a request is retried by the transport.

## New Media Design Flow

For a new image, video, or audio design, execute these steps in order. Use one tool call at a time and wait for each result.

1. Comprehend the user's intent and write a production-ready prompt in the user's language or the language requested by the user.
2. Create one text node with a concise summary title containing the enriched prompt with `canvas_create_node`.
3. Create one output node of the requested type with a concise summary title using `canvas_create_node`. Keep its prompt/content empty or set it to the enriched prompt as appropriate.
4. Pass every explicitly named input node id in `source_item_ids` when creating the Prompt and output nodes. The Canvas backend persists those source-to-target lines automatically. Also call `canvas_connect_nodes` when a semantic edge is needed that is not covered by those source ids, such as the Prompt-to-output edge. Do not create duplicate lines.
5. Generate only into the output node using the matching Canvas generation tool and its returned `canvas_item_id`.
6. Treat an accepted asynchronous task as started, not completed. Let the Go backend poll and persist the result.

For a new text-only request, create or update only the text node needed for the result. Do not create a pointless media node.

## Existing Node Flow

When `canvas_item_id` is present:

- Text/note: refine or replace only its text content. Preserve the previous content if the model or backend fails.
- Image: call `canvas_generate_image` with the selected node id, the enriched prompt, selected quality/ratio, and only the requested upstream image references.
- Video: call `canvas_generate_video` with the selected node id, exact duration (5-15 seconds), ratio, resolution, keyframes, soundtrack/voice-print roles, and caption script. Preserve first/intermediate/last frame roles.
- Audio: produce a ready-to-speak script first, then call `canvas_generate_audio` with the exact script and selected duration/model. Do not save a failure message as the script.
- Do not create a Prompt node or downstream node for an existing-node request unless the user explicitly asks for a new graph.

## Connections And Errors

- A line is persisted only through `canvas_connect_nodes` or a `source_item_ids` edge on Canvas node creation; never claim a connection exists after merely describing it.
- Do not create duplicate lines, self-loops, or connections to nodes outside the current Canvas.
- On tool failure, report a concise actionable error and preserve existing node content and media. Do not retry the same generation automatically.
- Do not expose provider errors containing secrets or internal URLs. The backend response is the source of truth for status.
