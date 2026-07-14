# Layout, Pinyin, And English Rules

Use for cover optimization, finished-page export, text-on-image, pinyin/Chinese/English trilingual text, and reference-layout migration.

Core principle: inner-page body text belongs to the deterministic layout layer. Short cover titles/subtitles may be generated inside the cover image when useful. All text must be readable, non-overlapping, and must not cover key image content.

## Default Inner-Page Mode: `full_bleed_caption_trilingual`

Use when the illustration is full-bleed or protagonist/key-object placement is uncertain.

- Keep the full illustration; do not crop just to imitate a reference.
- Put a translucent cream rounded caption bar in a low-information area, usually bottom.
- Text order: pinyin on top, Chinese in the middle, English below.
- Lay out pinyin and Chinese using character cells: each cell width is `max(Chinese glyph width, pinyin width) + safe gap`.
- English is a sentence translation, not per-character aligned. Center or left-align it as a stable text block.
- Keep page number separate from text; decorative dots/stars must not cover punctuation.
- Split long sentences into short lines. Prefer child-friendly concise English.

## Split Text / Art Mode: `split_text_art_trilingual`

Use only when the image prompt reserved a large text area.

- Do not force an already full image into left-text/right-art.
- Use only when text area is at least 32% of page width and the image subject is not cropped.
- Trilingual pages need more text space than Chinese-only pages.

## Trilingual Text Data

Store text structurally:

```markdown
Pinyin: xiǎo tù bǎ yuè guāng zhuāng jìn kǒu dài lǐ.
Chinese: [the finalized Chinese sentence]
English: Little Bunny tucked the moonlight into his pocket.
```

- Finalize Chinese first, then generate pinyin and English.
- Keep tone marks; resolve polyphonic characters by sentence meaning.
- English should read like a children's picture-book line, not mechanical translation.
- Keep character names consistent.
- Use Chinese as the primary anchor for image/text contract checks.

## Cover Typography

The cover may ask image generation to render a short title/subtitle directly.

- Title must be first visual layer, but do not pile up stickers, labels, or annotations.
- Use rounded, thick, child-friendly Chinese title lettering when Chinese is needed.
- Choose title color from the illustration palette.
- Use light stroke/shadow for readability; avoid metallic/3D text.
- Subtitle is smaller and in the same font family.
- Keep fixed safety spacing between title, subtitle, decorations, and protagonist face.
- If title generation repeatedly fails, switch to no-text cover plus deterministic overlay.

Prompt example:

```text
Generate clear Chinese cover title: 《Book Title》
Subtitle: For children starting kindergarten
Rounded bold Chinese picture-book lettering, soft edges, readable, naturally integrated with the illustration.
Text must be exact, no typos, no gibberish, no extra words. Leave enough empty title area; do not cover the protagonist face or key prop.
```

## Badges And Stickers

- Badge dashed circles, borders, and stars must not cross text.
- Keep text within about 70% of the badge safe region.
- Two lines max; use short copy.
- If text touches decoration, shorten the copy or reduce font size. Do not let lines cross text.

## Pinyin / English Checks

- Pinyin aligns vertically with the corresponding Chinese characters.
- Pinyin does not touch Chinese glyphs.
- Punctuation has spacing and does not stick to the previous character.
- Pinyin font size is about 40%-50% of Chinese size.
- English uses enough line height and does not touch the caption bar edge.
- English must cover the Chinese action without adding objects absent from the image.

## Required Export Checks

Check cover, first page, longest-text page, badge page, and final page:

- title/subtitle do not overlap
- badge borders/decorations do not cut text
- pinyin does not collide with Chinese/punctuation/page number
- English does not collide with other text or edges
- caption bar does not cover protagonist face, hands, key props, or key shadows
- all text stays inside safety margins

## Reference Layout Migration

Mature picture books often reserve text space during composition. Do not hard-force a layout after the image is already generated.

- For left-text/right-art or top-text/bottom-art layouts, specify text-zone ratio and subject placement in the image prompt.
- For existing full-bleed illustrations, use `full_bleed_caption_trilingual`.
- For new illustrations, `split_text_art_trilingual` is allowed only if the page plan explicitly defines text zone, subject zone, and visual evidence zone.
