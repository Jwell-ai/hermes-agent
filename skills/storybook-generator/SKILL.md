---
name: storybook-generator
description: Create children's picture books, storybook MVPs, page-by-page illustration prompts, consistent multi-page characters, bilingual/trilingual layouts, KDP-ready publishing notes, and complete story-to-image workflows. Use when the user asks for a children's picture book, flipbook, illustrated story, page plan, storybook prompts, character bible, or storybook QA. Build story rhythm, character continuity, visual style, and page prompts before image generation.
---

# Storybook Generator

Turn a rough story idea, short text, educational topic, or character concept into a low-cost picture-book MVP, then into consistent page-by-page images: topic framing, story arc, page rhythm, character/world/style bible, page-level image prompts, cover plan, sample layout checks, and QA.

## Reference Routing

Read only the references needed for the task:

- `references/story-structure.md`: story structure, page rhythm, text density, and age adaptation. Read before splitting a story into pages.
- `references/character-continuity.md`: character consistency, scene continuity, and visual anchors. Read before generating multiple pages.
- `references/visual-styles.md`: picture-book art directions and prohibited styles. Read before choosing style or writing prompts.
- `references/prompt-workflow.md`: page prompt template, generation order, file naming, and delivery format. Read when generating images.
- `references/layout-and-pinyin.md`: final page layout, cover typography, pinyin/Chinese/English layout, text-on-image rules. Read when exporting finished pages, adding text, pinyin, or English.
- `references/reference-corpus-lessons.md`: reusable lessons from picture-book reference corpora. Read when improving story quality or migrating a reference-book feel.
- `references/story-text-structure-lessons.md`: sentence functions, page-turn hooks, repetition, cumulative structure, and text mechanics. Read when writing or revising page copy.
- `references/commercial-publishing-workflow.md`: KDP/commercial MVP validation, cover/description/series workflow. Read for commercial publishing requests.
- `references/qa-checklist.md`: story coherence, child safety, image/text alignment, layout checks, and rework rules. Read before delivery.

## Workflow

## Alphart Edu Tool Contract

When this skill is used inside Alphart Edu, image generation is not performed
with generic local image tools or file-writing workflows. Use this skill to
prepare story structure, character continuity, visual evidence, QA, and
page-level image prompts, then call `canvas_create_storybook` or
`create_storybook`.

Required Alphart Edu behavior:

- Call `canvas_create_storybook` / `create_storybook` for storybook creation.
- Pass `generate_images: true`.
- Pass `aspect_ratio: "1:1"`.
- Pass an explicit `pages` array.
- For Alphart Edu physical pages, use this rhythm:
  - cover page: image page with `image_prompt`
  - back cover page: image page with `image_prompt`
  - inner left / odd story pages: image pages with `image_prompt`
  - inner right / even story pages: narration/text pages without `image_prompt`
- Do not call `generate_image`, `canvas_generate_image`, Write, Bash, local
  file tools, or HTML/export tools to create storybook images.
- If reference images are present, pass their `s3_object_name` values in
  `input_images`; do not convert them to base64.

The Alphart Edu backend will use its configured image model to generate and
store required storybook page images.

### 1. Decide The Delivery Mode

- If the user asks to "generate", "make", "create", or "turn this into a storybook", produce the storybook plan and proceed to page image generation.
- If the user only asks for a plan, storyboard, page prompts, or "no images", do not call image generation.
- If the user reports a problem with an existing book, enter repair mode. Identify the page, problem type, root cause, minimum repair action, and which old images can be kept.
- Defaults when the user does not specify: ages 3-8, no more than 12 story pages plus cover, square or landscape picture-book pages, warm hand-drawn style.

Repair mode should not rebuild the whole book by default. First report:

- page number and issue type
- whether the root cause is story structure, image/text contract, character anatomy, text rendering, or layout
- minimal repair action: revise text, revise prompt and regenerate `-v2`, repair page plan, or re-export layout
- which images remain valid and which need replacement

### 2. Build The Storybook Skeleton First

Use `references/story-structure.md`.

Normalize the input into:

- target reader age and reading context
- concrete topic scenario and parent/purchaser reason
- one-sentence theme and emotional keywords
- protagonist, desire, obstacle, turn, ending
- page count and narrative job of each page
- draft text per page; keep it short, concrete, and drawable
- page-to-page causal chain: previous state -> current trigger -> action/discovery -> next hook
- visual contract: every visible noun/action in the text must have matching visual evidence in the prompt

Do not ask AI to "write a children's book" directly. Use an MVP chain:

topic -> three-part outline -> page text -> character bible -> storyboard prompts -> cover plan -> sample layout -> pre-publishing QA.

For commercial requests, extend it:

concept validation -> story structure -> character/style bible -> page illustrations -> cover -> platform/KDP description -> series expansion -> layout export.

### 3. Build The Consistency Bible

Use `references/character-continuity.md`.

Create:

- fixed protagonist appearance
- supporting character appearance and relationship
- scene/world anchors
- prop anchors
- style anchors
- per-page continuity rules

For every page prompt, repeat the key fixed appearance details. Do not rely on "same character".

### 4. Choose One Visual Direction

Use `references/visual-styles.md`.

Prefer one of:

- warm hand-drawn picture book
- paper-textured playful hand-drawn
- clean whiteboard/simple line drawing story
- child-safe minimal character book
- folk/eastern hand-drawn style

Do not mix many styles in one book.

### 5. Generate Page By Page

Use `references/prompt-workflow.md`.

Each page prompt must include:

- page type and aspect ratio
- fixed character appearance
- current page action, setting, emotion, camera
- visual evidence required by the page text
- style/material/color/white-space rules
- negative constraints: no photo, 3D, poster, dense text, wrong text, watermark, copyrighted characters

Generate one image per page. Generate cover, inner pages, and back cover separately. Do not combine multiple pages into one image.

### 6. Text-On-Image Strategy

Inner-page body text should normally be added later by deterministic layout, not generated inside the image. This avoids misspellings and layout drift.

Default:

1. Generate illustrations without body text.
2. Render story text in HTML/PPT/PDF/canvas layout.
3. Use fixed text zones, speech bubbles, blank areas, or translucent caption bars.
4. Export finished page PNG/PDF if needed.

Cover text may be generated directly only when the title/subtitle is short. Check it character by character. If it fails repeatedly, use a no-text cover plus deterministic overlay.

### 7. QA And Rework

Use `references/qa-checklist.md` before delivery.

Check:

- story has a beginning, build, turn, and ending
- characters remain consistent
- every page has a clear action
- text and image match
- content is age-appropriate and safe
- no malformed hands, limbs, faces, or accidental extra objects
- text is readable and not overlapping
- cover, first page, longest text page, badge page, and last page all pass layout checks

If a page fails, identify problem type first:

- broken story: fix page plan causal chain
- text/image mismatch: revise text or regenerate image
- anatomy error: regenerate the page with explicit limb constraints
- text error: remove in-image text and use layout overlay

## Output Style

Planning output should be short but complete:

- title
- age
- page count
- protagonist
- style
- page-by-page plan

Generation output should report generated pages, where results are stored, and which pages need review. Avoid long theory unless the user asks.
