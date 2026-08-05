# Changelog

## Unreleased - 2026-08-04

- Added Canvas-only recovery for Anthropic-compatible relay streams that close without a final `message_stop`; Edu retains its existing streaming behavior.
- Canvas prompt values now override Canvas UI fallbacks for media duration, aspect ratio, resolution, quality, model, and generated-audio preference; Edu keeps its existing defaults.
- Canvas audio requests now preserve an explicit duration from the user's prompt while leaving Edu's legacy audio flow unchanged.
- Restricted Canvas video relay references to Canvas S3 object keys and exposed relay errors instead of retrying an unreachable model-supplied URL; Edu video behavior remains unchanged.
- Added Canvas-only recovery for image generation graphs so prompt and image nodes are materialized before the media relay runs.
- Ignored a selected text or note node when the model incorrectly returns it as an image execution target.
- Preserved the existing Edu image-generation path and routing behavior.
