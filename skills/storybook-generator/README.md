# Storybook Generator Skill

This skill turns a rough story idea into a structured picture-book MVP: story beats, page plan, character/style bible, page-level image prompts, layout rules, QA checks, and optional publishing notes.

## Why It Is More Than A Prompt

Most AI storybook workflows fail at the production layer, not the imagination layer:

- The story has no page-to-page causality and becomes a pile of pretty images.
- The protagonist drifts across pages: clothes, body shape, expression, props, and age change.
- Text mentions objects/actions that the image does not show.
- The image model writes body text, causing typos, gibberish, and layout failure.
- Cover, inner pages, bilingual/trilingual text, publishing description, and QA do not form one workflow.

`storybook-generator` makes the agent build the production chain first: story rhythm, character continuity, visual anchors, image/text contract, page prompts, deterministic layout, and QA.

## Capabilities

- Turn a topic, lesson, short text, or character idea into a picture-book MVP.
- Create a page plan with narrative function, page-turn hook, visual evidence, and draft text.
- Build a character and style bible before image generation.
- Write stable page-level prompts for image generation.
- Preserve visible objects, actions, hands, props, and scene anchors across pages.
- Add layout guidance for no-text illustrations, covers, pinyin, Chinese, English, and mixed-language pages.
- Check finished pages for story logic, illustration defects, text errors, and child safety.
- Extend a book concept into publishing notes, product positioning, and series expansion.

## Suitable Requests

- "Help me make a children's picture book."
- "Split this story into 12 picture-book pages."
- "Give me a character bible and page prompts."
- "Keep the same protagonist consistent across pages."
- "Make a Chinese/pinyin/English picture book."
- "Make a KDP-ready picture-book MVP."
- "Review this storybook for continuity and rework pages."

## Repository Structure

```text
storybook-generator/
  SKILL.md
  agents/
    openai.yaml
  references/
    story-structure.md
    character-continuity.md
    visual-styles.md
    prompt-workflow.md
    layout-and-pinyin.md
    reference-corpus-lessons.md
    story-text-structure-lessons.md
    commercial-publishing-workflow.md
    qa-checklist.md
```

## Design Philosophy

Picture books are systems.

A good page has a job. A good spread has a visual contract. A good character has repeated anchors. A good book has rhythm, restraint, escalation, and a final emotional turn.

This skill teaches the agent to treat storybook creation as a repeatable creative pipeline: editorial structure first, generation second, QA always.

