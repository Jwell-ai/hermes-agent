---
name: studio-plan
description: "Game Studio planning phase — Producer breaks GDD sections into epics and user stories with effort estimates and acceptance criteria."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, planning, sprint, epics, stories, producer, backlog]
    related_skills: [game-studio, studio-design, studio-dev]
---

# Studio · Planning Phase

**Active roles:** Producer (breakdown + estimates) · Lead Programmer (technical sizing)

## Goal
Convert a GDD section or milestone description into a prioritised backlog of epics and
stories, each sized and acceptance-tested, ready for a development sprint.

## Invocation
```
/studio plan <milestone-or-section>
# examples:
/studio plan combat system
/studio plan milestone-1 MVP
/studio plan sprint 3
```

## Process

### Step 1 — Input
Accept one of:
- A GDD section name (reads `design/<section>.md` if present)
- A milestone description in natural language
- An existing epic to split further

### Step 2 — Epic Breakdown
Group work into epics (1–2 week chunks). For each epic:
```
EPIC [N]: <title>
Goal    : One sentence — what player-visible outcome does this deliver?
GDD Ref : design/<section>.md (or N/A)
Stories : [list below]
```

### Step 3 — Story Cards
For each story:
```
STORY [epic.n]: <title>
As a <role>, I want <action> so that <outcome>.

Acceptance Criteria:
  - [ ] …
  - [ ] …

Effort  : XS (< 2h) · S (2–4h) · M (0.5d) · L (1d) · XL (2d+)
Priority: P1 (must) · P2 (should) · P3 (nice-to-have)
Depends : STORY [x.y] (if any)
```

### Step 4 — Scope Check
Producer flags:
- Stories with no acceptance criteria → block until defined
- XL stories → recommend splitting
- P1 stories blocked by P2/P3 → reorder dependencies
- Epics exceeding 2-week budget → split or defer to next milestone

### Step 5 — Output
Write the backlog to `production/backlog-<milestone-slug>.md`.
Print a summary table:

| Epic | Stories | Total Effort | P1 Count |
|------|---------|-------------|---------|
| …    | …       | …           | …       |

Offer to continue with `/studio dev <story>` to begin implementation.
