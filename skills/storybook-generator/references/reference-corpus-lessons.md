# Lessons From Picture-Book Reference Corpora

These are observations from classic picture-book references. Do not copy any one book. Extract reusable story, composition, and layout principles only.

## Reference Boundaries

- Old `.ppt` files may not open cleanly or expose images.
- Use patterns, not direct style copying: rhythm, page types, text-zone planning, action readability.
- Do not reuse copyrighted characters, original compositions, or original text.

## Story Rhythm

### Repetition With Escalation

For bedtime, fear of the dark, or timid-child topics, do not repeat one emotion. Use rounds:

1. A concrete little problem appears.
2. Protagonist performs a visible action.
3. A new discovery appears.
4. Next round is similar but slightly escalated.
5. In the end, the protagonist actively uses the learned method.

Example for fear of the dark:

see shadow -> check with flashlight -> find source -> name the shadow -> do a shadow experiment.

### Fixed Scene With Incremental Change

Reusing the same room, bed, or window is good for AI continuity.

- Change only one variable per page: light, prop, character position, shadow shape, emotion.
- End by echoing the opening scene with a changed emotional state.
- In an 8-page inner story, keep at least 5 pages in the same core location.

## Page-Type Rotation

Do not default to cinematic full pages. Rotate by narrative job:

- full scene: world, atmosphere, key turn
- whitespace action page: single action/expression
- small-panel sequence: practice/repetition/feedback
- object close-up: clue/cause/discovery
- echo page: compare beginning and ending

## Action Readability

In younger picture books, clear action matters more than complex lighting.

Every prompt must answer:

- What is the protagonist doing?
- What are they holding, and how many hands are visible?
- Where are their eyes looking?
- Which object explains the page text?
- How does this action lead to the next page?

## Plan The Text Area Early

Reference books often place text in white borders, sky, walls, page edges, or low-information zones.

- For left-text/right-art, prompt: "left 30%-35% blank text area, subject on the right".
- For existing full-bleed art, use full image plus bottom translucent caption bar.
- Pinyin needs larger safety space; do not place pinyin over complex backgrounds.

## Cover Lessons

The cover must communicate title, protagonist, conflict/value:

- Title is first visual layer.
- Protagonist and core prop are visible.
- Parent can identify the problem solved: darkness, bedtime, shyness, separation, etc.
- Badges are supplementary; they must not cover key image content.

## Large Text Inside Image

Short words or sound effects can be visual rhythm, but this is not the default.

- Body text should normally be laid out after image generation.
- Allowed in-image text: cover title, door signs, sound words, short commands.
- If using big handwritten visual text, keep a no-text or layout-overlay backup.

