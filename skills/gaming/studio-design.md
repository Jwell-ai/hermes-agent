---
name: studio-design
description: "Game Studio design phase — Game Designer drafts GDD sections; Technical Director validates feasibility; Directors gate-check before sign-off."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, design, gdd, architecture, technical-director, game-designer]
    related_skills: [game-studio, studio-brainstorm, studio-plan]
---

# Studio · Design Phase

**Active roles:** Game Designer (authoring) · Technical Director (feasibility) · Creative Director (pillar gate)

## Goal
Produce or extend a GDD section — mechanics, systems, UI/UX, narrative, or architecture —
and get a dual gate-check (creative + technical) before handing off to planning.

## Invocation
```
/studio design <topic>
# examples:
/studio design combat system
/studio design economy
/studio design main menu UX
/studio design save system architecture
```

## Process

### Step 1 — Scope the Section
Identify which GDD section this falls under:
- **Mechanics** — verbs, interactions, physics rules
- **Systems** — economy, progression, inventory, AI behaviour
- **UI/UX** — flows, HUD, menus, accessibility
- **Narrative** — story beats, dialogue system, world-building
- **Architecture** — data flow, module boundaries, persistence

### Step 2 — Draft the Section
Write the GDD section in this structure:
```markdown
## [Section Title]

### Overview
One-paragraph summary of what this system does and why it exists.

### Player-Facing Behaviour
Bullet list of what the player can see, feel, and do.

### Rules & Constraints
Numbered invariants the implementation must never violate.

### Data Model (if applicable)
Key entities and relationships (plain text or mermaid diagram).

### Edge Cases
List known edge cases and their intended resolution.

### Open Questions
Items requiring a decision before implementation begins.
```

### Step 3 — Technical Gate (Technical Director)
Assess:
- Is the proposed approach implementable in the chosen engine within sprint budget?
- Are there known engine limitations that require a design change?
- Does the data model introduce O(n²) or worse hot paths?

Return: `[TECH-GATE]: APPROVE | CONCERNS | REJECT` + rationale.

### Step 4 — Creative Gate (Creative Director)
Assess using the Director Decision Filter (see game-studio skill).
Return: `[CREATIVE-GATE]: APPROVE | CONCERNS | REJECT` + rationale.

### Step 5 — Write to Disk
If both gates pass, write the section to `design/<section-slug>.md`
(create `design/` if it does not exist). Confirm the write to the user.

Offer to continue with `/studio plan` to generate epics from this GDD section.
