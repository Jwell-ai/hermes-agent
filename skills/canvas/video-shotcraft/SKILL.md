---
name: canvas-video-shotcraft
description: Direct a Canvas video with video-shotcraft cinematic product-video shot-card grammar. Use only when the user explicitly invokes [skill:video-shotcraft], names video-shotcraft, or asks for shot-card or shot-recipe direction.
---

# Canvas Video Shotcraft

Use this workflow only for generation or regeneration of the selected Canvas
video node. It adapts video-shotcraft's product-video directing principles to
Canvas and Seedance. Do not use Remotion, a local project, browser capture,
external CLIs, the Ink Press template, or any bundled third-party assets.

## Scope And Inputs

- This is a Canvas-only workflow. Work on the selected video node; never create
  a replacement node or alter a connected node unless the user explicitly asks.
- Connected text/note nodes establish the product story, feature, action, and
  ending. Connected image nodes establish product framing and visual continuity.
  Connected audio nodes keep their Canvas role: soundtrack, background music,
  or voice print.
- Treat user-provided `@` roles as authoritative. Use `@Image as first frame`,
  `@Image as intermediate frame`, and `@Image as last frame` only in their
  named roles. Never invent media, IDs, products, screens, or UI details.

## Shotcraft Direction

Turn the request into one compact, production-ready video prompt before
dispatching. The result should show a real product or product concept clearly,
not a decorative montage.

1. Pick one feature or story beat as the shot's protagonist. State the product
   state before the action, the single clear action, and the resolved ending.
2. Preserve the supplied product visual language: layout density, typography,
   palette, materials, lighting mood, and the spatial direction established by
   connected frames. Do not introduce an unrelated promo aesthetic.
3. Choose exactly one camera grammar: framing, lens feeling, one deliberate
   move, speed, and a final hold. Avoid competing moves, random motion, or
   repeated reveal effects.
4. Pace the action to the requested duration: 5 seconds is setup -> action ->
   hold; 6-9 seconds can include one transition; 10-15 seconds may use up to
   three concise timed beats. Leave the last beat readable rather than filling
   every second with movement.
5. When the user asks for a shot card or recipe, translate it into a visual
   mechanism such as card-deal, camera push, orbit, match cut, type reveal, or
   controlled transition. Describe the mechanism; do not claim access to the
   original demo implementation or gallery assets.
6. Do not put dialogue, captions, subtitles, or on-screen text instructions in
   the provider prompt. Canvas owns TTS and SRT separately. If a soundtrack or
   BGM node is attached, preserve it and disable provider-generated audio;
   otherwise allow only ambient provider audio.

## Prompt Shape

Write in this order:

`product/subject and continuity; setting and visual language; one action;
camera framing and movement; timed progression; final hold; audio constraints.`

For a longer video use compact timing such as `0-3s`, `3-7s`, and `7-10s`.
Be concrete about what becomes visible and when. Keep the prompt concise enough
for the selected video provider.

## Dispatch

Call `canvas_generate_video` exactly once with the selected node, connected
references, selected model, ratio, resolution, and exact requested duration.
Canvas owns storage, billing, task polling, captions, and media persistence.
Report a concrete tool error without overwriting existing node content.

## Provenance

Adapted from Vincentwei1021/video-shotcraft at commit
`d4915443232e89527fdc9d7e79f132ba411fc440` (MIT):
https://github.com/Vincentwei1021/video-shotcraft
