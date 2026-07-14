# Picture-Book Story Structure

## Default Parameters

- Reader: ages 3-8 unless the user specifies otherwise.
- Length: default cover + up to 12 inner pages; short stories can be 6-8 pages, complex stories can be 16-24 pages.
- Text per page: ages 3-5 use 0-18 words/characters; ages 6-8 use 10-35. Avoid long abstract sentences.
- Narrative unit: each page carries exactly one action, discovery, emotional change, or choice.

## MVP Flow

Do not directly generate the whole book. Use a low-cost MVP path:

1. Topic: tie it to a real child scenario.
2. Three-part outline: opening problem, attempts/misunderstanding, natural understanding.
3. Page text: write by page, with text and image suggestion.
4. Character profile: fixed appearance, clothes, actions, visual elements that cannot change.
5. Image/text contract: list visible nouns/actions in the page text and verify they can be drawn.
6. Storyboard prompts: one prompt per page.
7. Cover plan: make the problem/value clear at a glance.
8. Sample layout: unified size, fonts, safety margins, text does not cover key image content.
9. Self-check: story, age fit, consistency, image defects, copyright risk, and parent purchase reason.

## Topic Before Story

Ground the topic in a situation children understand:

- a 3-year-old refusing to brush teeth
- a 5-year-old afraid of the dark
- a 6-year-old anxious about kindergarten
- a 7-year-old who hates losing
- an older sibling feeling ignored after a new baby

Topic output should include:

- title
- child-relatable scenario
- parent purchase reason
- educational value

If the user gives a broad theme, offer 10-20 possible directions. If the user gives a specific scenario, go straight to outline.

## Five-Sentence Skeleton

Compress the input into five sentences:

1. Who the protagonist is and what they want.
2. What interesting rule or limit exists in the world.
3. The first attempt and where it fails or is misunderstood.
4. The key discovery that resolves the problem through action, not preaching.
5. A warm ending with a visible echo.

Children's picture books should be action-driven, not lesson-driven. Translate themes such as bravery, sharing, persistence, environment, AI, or learning into visible child actions.

Avoid preaching. The message must grow from the plot instead of being announced by narration.

## Continuity Rules

Page text is not a set of unrelated posters. Every page must answer:

- What state/question did the previous page leave?
- What visible action happens on this page?
- What does the protagonist learn, or how does emotion change?
- Why does the next page naturally happen?

If removing a page does not affect the story, merge or rewrite it.

For psychological topics such as fear of the dark, separation anxiety, or losing gracefully, do not repeat the same emotion. Every 2-3 pages the protagonist needs visible progress: avoiding -> observing -> checking -> naming -> acting independently.

## Image/Text Contract

Before writing page text, decide whether it can be drawn:

- Concrete objects, body actions, directions, and quantities in the text must appear in the image prompt.
- Do not put unstable visual details into the core sentence.
- If the text says "a little hand on the wall", the prompt must specify "a small hand-shaped shadow on the wall, five round fingers, cast by leaves near the window"; otherwise rewrite the text.
- Keep each page to one key discovery for younger readers.
- Text cannot rescue an unclear image. If the picture cannot be understood, do not explain it away with text.

## Page Rhythm

### 8 Inner Pages

1. Introduce protagonist and everyday world.
2. Desire or small trouble appears.
3. First attempt.
4. Mistake or misunderstanding deepens.
5. Help, clue, or new view appears.
6. Protagonist chooses independently.
7. Problem is gently resolved.
8. Ending echo; the world has changed a little.

### 12 Inner Pages

- 1-2: protagonist, world, desire
- 3-4: trouble appears, first attempt
- 5-6: failure, confusion, emotional low point
- 7-8: clue/helper, key understanding
- 9-10: protagonist acts, resolution begins
- 11: result lands
- 12: ending echo or humorous aftertaste

### 16 Inner Pages

Add more try-feedback rounds, not more major characters. Prefer emotional depth in one core scene over a parade of events.

## Rewrite Principles

- Replace abstractions with concrete objects: turn "anxiety" into "a tiny alarm clock ticking in the pocket."
- Use repeated sentence patterns, but change a small variable every time.
- Every page sentence must be drawable.
- Avoid adult moral summary endings.
- Do not write "from then on, he understood the lesson."

## Page Plan Format

```markdown
| Page | Narrative job | Previous cause | Sentence function | Visual action | Visual evidence | Pinyin | Chinese | English | Next hook |
|---|---|---|---|---|---|---|---|---|---|
```

Generate page plans in this order:

1. narrative function
2. sentence function and page-turn hook
3. primary page text
4. pinyin/translation if needed
5. visual action and evidence

Do not write prose first and split later.

## Cover Plan

The cover must make the problem/value visible:

- composition
- protagonist placement
- background elements
- title layout suggestion
- color palette
- prohibited elements

The cover gets the click; the inner pages keep the reader.

