# Page-By-Page Prompt Workflow

## Project Directory

When working in a workspace, use:

```text
storybook/<book-slug>/
  00-storybook-brief.md
  01-character-style-bible.md
  02-page-plan.md
  03-cover-plan.md
  04-sample-book-notes.md
  prompts/
  images/
```

Use a short lowercase English/pinyin slug such as `little-rain-fox`. Do not overwrite old projects.

## Generation Order

1. Generate the cover or protagonist reference first to calibrate the character.
2. Before inner pages, finish the page causal chain and image/text contract.
3. Generate pages 1-3, then check character, style, text contract, and anatomy stability.
4. Continue only after the first pages are stable.
5. Generate back cover or title page last.

If the first three pages are unstable, repair the character bible before batch generation.

## Single-Page Prompt Template

```text
Generate one children's picture-book illustration, [aspect ratio].

Book/project: [title]
Page: [cover / page N]
Reader: [age range]

Fixed character:
[repeat the key appearance, clothing, accessories, expression baseline from the character bible]

Current page scene:
[one action, setting, emotion, prop, and camera for this page only]

Image/text contract:
[visual objects/actions/shapes from page text that must appear in the image]

Limb constraints:
[visible hands/feet count, left/right hand roles, occlusion; e.g. only two visible hands, right hand holds lamp, left hand hugs teddy, no third hand]

Page text:
[Cover: exact title/subtitle if text must appear. Inner page: preferably no in-image body text.]

Style:
[main material, line, color, whitespace, age-appropriate mood]

Continuity:
Keep character appearance, clothing, accessories, and palette consistent with previous pages. Do not change age, species, hairstyle, fur, clothing color, or core prop.

Negative constraints:
No photo, no 3D, no commercial poster, no dense UI, no watermark, no logo, no copyrighted characters, no horror/adult content, no dense text, no typo/gibberish, no extra arms, no third hand, no extra fingers, no fused limbs.
```

## Storyboard Prompt Ingredients

Every page prompt must include:

- character profile
- current page sentence or plot
- what the protagonist is doing
- where the scene is
- emotion
- camera distance
- color mood
- visual evidence for page text
- hand/foot count and prop division
- whether image text is forbidden or limited
- consistent picture-book style

## Aspect Ratio

- `1:1`: social previews, younger readers, single-page stories, printer-friendly square pages.
- `4:3` or `16:9`: screen reading, story videos, PPT-like books.
- `3:4`: phone reading.
- Print spreads: first stabilize single pages, then consider spread composition.

## Text Strategy

Cover title/subtitle may be generated directly if short. The prompt must include exact text and require clarity, no typos, no extra text. Check afterward character by character.

Inner-page body text should usually be added later:

1. Generate no-text illustration.
2. Build a page layout in HTML/PPT/PDF/canvas.
3. Add text in a fixed text area, bubble, whitespace, or translucent caption bar.
4. Export finished page PNG/PDF.

This keeps Chinese fonts, pinyin, and translations correct, and lets the user revise copy without regenerating art.

## Pre-Generation Check

- Does the page have only one action/discovery?
- Are all key text objects/actions in the current scene and visual contract?
- Is the protagonist doing too many actions?
- For characters, are visible hands/feet and left/right roles specified?
- Is text space needed?

## Post-Generation Check

Check four things immediately:

- Image/text alignment: every key object/action is visible.
- Anatomy: no extra hands, fingers, misplaced limbs.
- Continuity: clothing, prop, room, and style remain stable.
- Emotional progression: the page moves the story forward.

If it fails, do not continue batching. Rewrite that page prompt and save as `-v2`.

## Naming

```text
images/00-cover.png
images/01.png
images/02.png
images/01-v2.png
prompts/page-00-cover.md
prompts/page-01.md
```

Do not overwrite old versions.

## Delivery Format

Before generation:

```markdown
## Storybook Plan
- Title:
- Age:
- Pages:
- Protagonist:
- Style:

## Page Plan
| Page | Visual | Pinyin | Chinese | English |
```

Sample-book stage:

```markdown
## Sample Layout Check
- Page size:
- Fixed font:
- Image safety margin:
- Text over image:
- Trilingual order:
- Cover/title/copyright/body pages:
- PDF export notes:
```

After generation:

```markdown
Generated:
- Cover: images/00-cover.png
- Page 1: images/01.png

Needs review:
- Page 3 scarf color drifted slightly; recommend regenerating v2.
```

