---
name: studio-dev
description: "Game Studio development phase — Lead Programmer delegates implementation to the correct specialist (Godot/Unity/Unreal/engine-agnostic) with code review gates."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, dev, implementation, programmer, godot, unity, unreal, code-review]
    related_skills: [game-studio, studio-plan, studio-qa]
---

# Studio · Development Phase

**Active roles:** Lead Programmer (delegation + review) · Engine Specialist · Relevant sub-system specialist

## Goal
Implement a story or sub-system correctly for the detected engine, following the project's
coding standards, then gate-check the output before marking the story done.

## Invocation
```
/studio dev <story-or-sub-system>
# examples:
/studio dev STORY 1.3
/studio dev player movement controller
/studio dev save/load system
```

## Engine Specialist Routing

| Detected | Active Specialist |
|----------|------------------|
| `project.godot` | Godot 4 · GDScript or C# depending on existing files |
| Unity `Assets/` | Unity · DOTS if `com.unity.entities` in manifest |
| `*.uproject` | Unreal Engine 5 · Blueprint or C++ per project convention |
| None | Engine-agnostic (pseudocode + architecture) |

## Coding Standards (enforced by Lead Programmer)

**All engines:**
- No magic numbers — all tunable values in a data resource / ScriptableObject / DataAsset
- No game state in UI components
- Input handling separated from game logic
- No synchronous disk I/O on the main thread

**Godot:** follow `res://` path conventions; signals over direct method calls for decoupled systems.  
**Unity:** MonoBehaviour for scene glue only; core logic in plain C# classes; Addressables for runtime assets.  
**Unreal:** prefer Gameplay Ability System for abilities; Blueprint for designers, C++ for performance-critical paths.

## Process

### Step 1 — Story Clarification
Read the acceptance criteria. If any criterion is ambiguous, ask before writing code.

### Step 2 — Implementation Plan
Outline the approach in 3–5 bullet points before writing any code. Confirm with the user.

### Step 3 — Implementation
Write the code. For each file:
- Respect existing naming conventions (detect from adjacent files)
- Keep functions ≤ 40 lines; extract helpers for anything longer
- One public responsibility per class/node

### Step 4 — Self-Review (Lead Programmer gate)
Check the output against:
- [ ] All acceptance criteria satisfied?
- [ ] No hardcoded magic numbers?
- [ ] No game state leaked into UI?
- [ ] Engine-specific best practices followed?
- [ ] Any TODO left behind? (block story done if yes)

### Step 5 — Mark Done
If the review passes, print:
```
✅ STORY [n.n] DONE
Files changed: …
Next recommended: /studio qa <area>
```
