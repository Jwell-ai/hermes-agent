---
name: canvas-video-shot-breakdown
description: Analyze a completed Canvas video shot by shot for directing and critique. Use when the user asks for shot breakdown, shot-by-shot analysis, 拉片, 分镜分析, 镜头分析, or [skill:video-shot-breakdown].
---

# Canvas Video Shot Breakdown

Use this workflow only for analysis of an existing completed Canvas video. It is
adapted from the supplied `video-shot-breakdown` skill for Hermes' native
`video_analyze` tool.

## Required Behavior

1. Use the Canvas video URL supplied in the system context. Do not invent a URL,
   download the asset locally, create an HTML file, or call a generation tool.
2. Call `video_analyze` exactly once. Ask for an ordered shot list with timestamps,
   visual content, camera language, lighting, sound, editing, directorial intent,
   and practical teaching notes.
3. Return exactly one `<canvas-shot-breakdown>` block containing a JSON object.
   Canvas persists and renders this object as the shot-breakdown template; do
   not add prose before or after the block. The object must be shaped as:

   ```json
   {
     "meta": {
       "video_name": "string",
       "total_duration": "string",
       "resolution": "string",
       "fps": "string",
       "codec": "string",
       "style": "string",
       "shot_count": 1
     },
     "shots": [{
       "shot_num": 1,
       "start_time": "00:00:00",
       "end_time": "00:00:05",
       "duration": "5s",
       "visual_desc": "string",
       "content_analysis": "string",
       "shot_size": "string",
       "cam_angle": "string",
       "cam_motion": "string",
       "focal_length": "string",
       "depth_of_field": "string",
       "special_technique": "string",
       "light_type": "string",
       "light_direction": "string",
       "light_quality": "string",
       "color_tone": "string",
       "music": "string",
       "sfx": "string",
       "voice_type": "string",
       "voice_content": "string",
       "editing": "string",
       "director_intent": "string",
       "teaching_notes": "string",
       "emotion_tag": "string"
     }],
     "lessons": ["string", "string", "string"]
   }
   ```

   Use `not clearly identifiable` for unsupported observations. Do not use
   Markdown fences inside the block.
4. Use only observations supported by the video. Mark audio, dialogue, or an exact
   lens choice as `not clearly identifiable` when the tool cannot establish it.
5. Include three concise, actionable production lessons in `lessons`. Never
   generate, replace, connect, or modify a Canvas node during analysis.

## Analysis Lens

For each shot, assess where visible: shot size, camera angle, camera movement,
framing, focal-length impression, depth of field, special technique, light
source/direction/quality, colour tone, music, SFX, voice, edit/transition,
emotion, director intent, and teaching notes.

For AI-generated video, describe lighting as virtual/CG when appropriate and
camera movement as simulated rather than claiming physical production facts.
