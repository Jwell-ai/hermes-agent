---
name: canvas-gpt-image2-keyframes
description: Create or refine strong Canvas image-node keyframes using GPT Image 2 prompt craft and the Canvas image relay.
---

# Canvas GPT Image 2 Keyframes

Use this skill for Canvas image-node generation and image refinement. It guides
the prompt only. Never invoke an external CLI, read API keys, write local files,
or bypass Canvas storage, billing, or the selected image model.

## Keyframe Workflow

1. Infer the image's production role: hero keyframe, opening frame, ending
   frame, character reference, product reference, setting, or transition cue.
2. Preserve connected reference-node invariants named by the user: identity,
   silhouette, wardrobe, object shape, layout, palette, lighting direction, and
   composition. Do not silently change them.
3. Structure the prompt as: setting -> primary subject -> required details ->
   composition/camera -> lighting/material -> style/finish -> exclusions.
4. Choose the requested aspect ratio before describing composition. Keep one
   clear focal subject and use supporting details sparingly.
5. For text that must appear in an image, quote it exactly and keep it short.
6. For edits, describe what must remain unchanged before describing the change.

## Video-Aware Keyframes

When the image will feed a video node, optimise it for continuity: readable
silhouette, stable subject placement, coherent perspective, uncluttered motion
space, and lighting that can transition into the requested next frame. A
keyframe is a visual constraint, not a complete video storyboard.

## Dispatch

Call `canvas_generate_image` exactly once for the selected image node using its
selected model, quality, ratio, and upstream references. The Canvas relay owns
S3 persistence, generation history, and billing.

## Upstream

Adapted for Canvas from `wuyoscar/GPT-Image2-Skill` at commit
`ecc9c5420c265f6677edc5f4d255bca02497ef71` (MIT):
https://github.com/wuyoscar/GPT-Image2-Skill
