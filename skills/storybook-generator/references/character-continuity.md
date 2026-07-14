# Character And World Continuity

## Write The Character Bible First

Use fixed fields for each important character:

```markdown
## Protagonist
- Name:
- Species / age impression:
- Role:
- Personality:
- Body silhouette:
- Facial features:
- Hair / fur:
- Fixed clothing:
- Fixed accessories:
- Common expression:
- Movement habits:
- Signature action:
- Hand / limb rules:
- Illustration style:
- Must not change:
```

The "must not change" field is the core of cross-page consistency: yellow raincoat, red boots, blue bow on the left ear, round black eyes, star backpack, etc.

For human/anthropomorphic characters, define hand/limb rules:

- normal two hands, two arms, two legs
- common prop handling, e.g. right hand holds lamp, left arm hugs teddy
- which limbs are hidden when blocked
- prohibited: third hand, extra fingers, fused arms, limbs from wrong positions, one hand holding two far-apart objects

## Visual Anchors

Each book should have three anchor types:

- character anchors: clothing, accessories, silhouette, colors
- scene anchors: blue mailbox, crescent tree, classroom plant by the window
- prop anchors: glowing pebble, torn map, red pencil, silver bell

Anchors should be few enough to repeat, but visually clear enough to preserve.

## Repeat Key Fields In Every Prompt

Do not write:

```text
the same little girl keeps walking
```

Write:

```text
Doudou, a 6-year-old girl with a round face and straight black bangs, wearing a yellow raincoat, red boots, and a star backpack, walks along the wet path
```

For animal characters, fix body shape, ears, tail, fur, scarf, backpack, and color.

When characters interact with props, specify hand use:

```text
Doudou shows only two hands: right hand holds the glowing bunny lamp, left hand hugs the teddy bear to her chest; no third hand, no extra fingers, no fused arms.
```

To reduce anatomy errors:

- use medium or half-body framing
- avoid three simultaneous actions
- split actions across pages
- place some props on a table/floor instead of in hands

## Camera Continuity

- Cover: protagonist + core prop + world mood.
- Early pages: medium/wide shots establish setting.
- Emotion pages: close or half-body shots show expression/action.
- Action pages: medium/wide shots show what the protagonist does.
- Ending: echo the opening composition with changed emotional state.

Vary camera distance, but do not wildly change character proportions or material style.

## Control Variables

Rework only one major variable at a time:

- character drift: strengthen character bible, simplify scene
- scene drift: repeat scene anchors, reduce new props
- style drift: reduce style words to the main material/style
- text errors: remove in-image text and use layout overlay
- anatomy errors: specify visible limbs, hand roles, occlusion, then regenerate the current page

## Multi-Character Limit

For younger children, default to one protagonist plus 0-2 supporting characters. More than three major characters hurts consistency and story clarity. Merge character functions in ensemble stories.

