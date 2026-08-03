---
name: canvas-seedance2-video-director
description: Direct a polished Seedance video from Canvas text, image, audio, and frame-reference nodes. Use before canvas_generate_video for Canvas video requests.
---

# Canvas Seedance 2 Video Director

Use this skill only for Canvas video-node generation. It is an internal prompt
craft workflow: do not call external CLIs, do not expose internal references,
and do not create extra Canvas nodes unless the user explicitly requests one.

## Inputs

Treat the Canvas context as authoritative:

- Connected text/note nodes are visual narrative direction. Canvas owns approved
  voiceover and caption rendering outside Seedance.
- `@Image as first frame` anchors the opening composition.
- `@Image as last frame` anchors the closing composition.
- `@Image as keyframe`, `@Image as intermediate frame`, or an unqualified image
  is a visual reference. Intermediate frames guide the transition; they are not
  a provider-guaranteed timecode.
- Connected audio nodes retain their supplied role: soundtrack, background
  music, or voice print.

Never invent an asset, an upstream node, an ID, a provider, or a model. Do not
replace the selected video node with a new node.

## Directing Workflow

Convert the user brief into one compact production prompt before generation.
Keep the work internal; the user-facing response should only confirm submission
or report a concrete error.

1. Identify the story beat: subject, objective, action, change, and ending.
2. Preserve continuity: character, wardrobe, prop, environment, time of day,
   colour palette, and spatial direction across supplied frame references.
3. Choose one deliberate camera grammar: framing, lens feeling, movement,
   speed, focus transition, and shot continuity. Avoid a list of conflicting
   camera moves.
4. Pace to duration: for 5 seconds use one clear action and one camera move;
   for 6-9 seconds use setup -> action -> resolution; for 10-15 seconds use
   short timed beats with a clear ending. Do not pad with repetitive motion.
5. State lighting, material, atmosphere, grade, and realism/stylization only
   when they materially improve the intended result.
6. Do not add dialogue, spoken words, captions, subtitles, or text overlays to
   the Seedance prompt. Canvas renders the approved TTS voiceover and SRT itself.
7. When a soundtrack/BGM reference is attached, preserve it and disable
   provider-generated audio. When no soundtrack/BGM reference exists, allow the
   provider to generate ambient audio only.

## Prompt Shape

Write a concise cinematic instruction in this order:

`subject and continuity; setting and lighting; action; camera/framing/motion;
temporal progression; visual finish; ambient sound constraints.`

For longer clips, use compact segments such as `0-3s`, `3-7s`, and `7-10s`.
Mention first and last frame constraints in prose only when those references
exist. Do not promise exact interpolation from an intermediate reference.

## Dispatch

Call `canvas_generate_video` exactly once with the selected Canvas video node,
its supplied image/audio references, the selected model, ratio, resolution, exact
requested duration, and a concise ready-to-speak `caption_script`. The caption
script is used by Canvas to generate the voiceover and SRT; it is separate from
the visual video prompt. The Canvas relay owns storage, task polling, billing,
and provider-specific media roles.

## Upstream

Adapted for Canvas from `dexhunter/seedance2-skill` at commit
`e06c7c63a766d623004a2807881c30685ce517af` (MIT):
https://github.com/dexhunter/seedance2-skill
