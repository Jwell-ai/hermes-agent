# Changelog

## Unreleased - 2026-08-04

- Restricted Canvas video relay references to Canvas S3 object keys and exposed relay errors instead of retrying an unreachable model-supplied URL; Edu video behavior remains unchanged.
- Added Canvas-only recovery for image generation graphs so prompt and image nodes are materialized before the media relay runs.
- Ignored a selected text or note node when the model incorrectly returns it as an image execution target.
- Preserved the existing Edu image-generation path and routing behavior.
