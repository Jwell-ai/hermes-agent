# Storybook QA Checklist

## Story Checks

- Protagonist has a clear desire or small trouble.
- Each page carries one action or emotional change.
- Pages have causality; this is not a random illustration set.
- Every page has: previous state -> current action/discovery -> next hook.
- Protagonist shows visible progress every 2-3 pages.
- Ending has a warm change or humorous echo.
- No adult preaching, slogans, or abstract concept piles.
- A parent can state the concrete problem the book solves.

## Age Fit And Safety

- No scary, gory, humiliating, sexualized, or adult imagery.
- Conflict is gentle and not solved by punishment or violence.
- Expressions are readable for young readers.
- Disease, disaster, or danger topics are protective and non-sensational.

## Visual Consistency

- Protagonist age, species, hairstyle/fur, clothes, and accessories remain stable.
- World and color anchors remain stable.
- Material/style is consistent across pages.
- Camera varies without abrupt proportion changes.
- Core props do not vanish or change color without reason.
- No strange hands, feet, limbs, props, or sudden face changes.
- Interaction pages must check hands: no third hand, no extra fingers, no fused arms.

## Single-Page Readability

- The page action is understandable without text.
- Key shapes, objects, and actions mentioned in text are visible.
- If not visible, revise text or regenerate; do not use text to explain an absent image.
- Subject is clear and background does not compete.
- Composition leaves space for text layout.
- Emotion/action matter more than decoration.
- Avoid too many tiny props for younger readers.

## Text Checks

- Body text is normally added by deterministic layout, not image generation.
- In-image text is minimal: cover title, labels, sound words.
- In-image text has no typos, gibberish, or missing characters.
- If image text is unstable, regenerate with no in-image text.
- Overlay text, pinyin, badges, dotted circles, page numbers, and decorations must not overlap.
- No copyright, platform, QR code, watermark, or logo in image.

## Sample Book / Publishing Checks

- Page size is unified.
- Fonts are fixed and consistent.
- Text does not cover protagonist or key actions.
- Cover title, subtitle, badge text, and decoration lines have safety spacing.
- Pinyin aligns with Chinese and does not touch glyphs.
- Images have safety margins.
- Cover, title page, copyright page, and body pages are complete when needed.
- PDF export spot-checks cover, first page, middle page, and final page.
- No copyright-risk characters or obvious imitation of living artists.

## Rework Priority

1. Character inconsistency: must fix.
2. Age-inappropriate or unsafe content: must fix.
3. Limb/anatomy errors: regenerate current page.
4. Image/text mismatch: revise text or regenerate until aligned.
5. Unclear page action: fix.
6. Text gibberish or layout overlap: remove in-image text or re-export layout.
7. Broken story causality: repair page plan before regenerating pages.
8. Minor style drift: optional if the book still reads consistently.

## Common Failures

### Text Says It, Image Does Not Show It

- Do not keep the page as-is.
- If the image is good but different, revise text to match.
- If the text is plot-critical, rewrite prompt with explicit visual evidence and regenerate `-v2`.

### Extra Hands / Feet / Fused Limbs

- Regenerate; do not hide with cropping/text.
- Reduce actions and props.
- Specify exactly: two visible hands, right hand does A, left hand does B, no third hand.

### Story Feels Like Loose Illustrations

- Pause image generation.
- Repair the page plan's previous-cause and next-hook columns.
- Merge repeated emotion pages.
- Add protagonist verification, choice, and action pages.
- Regenerate only affected pages.

