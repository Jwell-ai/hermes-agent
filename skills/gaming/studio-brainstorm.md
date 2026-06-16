---
name: studio-brainstorm
description: "Game Studio ideation phase — Creative Director + Game Designer generate game concepts, validate core fantasy, and produce a seed GDD outline."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, brainstorm, concept, ideation, creative-director, game-designer]
    related_skills: [game-studio, studio-design]
---

# Studio · Ideation Phase

**Active roles:** Creative Director (vision gate) · Game Designer (concepts)

## Goal
Generate 3 distinct game concepts for the user's prompt, evaluate them against the MDA
framework, then produce a ranked recommendation with a seed GDD outline for the winner.

## Process

### Step 1 — Clarify Context
Ask (if not already known):
- Target platform (PC / mobile / console / web)
- Desired play session length (< 5 min · 20 min · 1 hr · 4 hr+)
- Tone (dark · whimsical · realistic · abstract)
- Solo or multiplayer
- Any hard constraints (budget, engine, team size)

### Step 2 — Generate Concepts
Produce **3 concepts**, each with:
```
CONCEPT [N]: <title>
Core Fantasy : What can the player BE or DO uniquely?
Unique Hook  : "It's like X, AND ALSO Y"
Genre        : <primary> + <secondary>
Loop         : Core loop in one sentence
MDA Target   : Top 2 aesthetics (Sensation / Fantasy / Narrative / Challenge /
               Fellowship / Discovery / Expression / Submission)
Risk         : Biggest design or technical risk
```

### Step 3 — Director Gate
Creative Director evaluates each concept:
- Does it have a falsifiable core fantasy?
- Are the MDA targets achievable in stated constraints?
- Is there a clear competitive differentiator?

Label each: `✅ VIABLE` · `⚠️ CONCERNS` · `❌ REJECT` with one-line rationale.

### Step 4 — Winning Concept Outline
For the top-ranked viable concept, produce a seed GDD outline:
```markdown
# [Game Title] — Seed GDD

## Pillars (3–5 max)
1. …
2. …

## Core Loop
…

## Win / Fail States
…

## Scope Estimate
MVP features: …
Cut-list (v2+): …

## Open Questions
- …
```

Offer to continue with `/studio design` to expand this into a full GDD.
