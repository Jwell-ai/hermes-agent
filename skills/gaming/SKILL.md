---
name: gaming
description: "Game development skills for Alphart Edu: Game Studios workflow, design, development, planning, QA, and release guidance."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, gaming, game-studio, game-dev, design, qa, education]
    related_skills: [game-studio, studio-brainstorm, studio-design, studio-plan, studio-dev, studio-qa, studio-release]
---

# Gaming Skills

Use this skill for playable educational game generation and review.

This package exposes the existing gaming skill files in this directory. For
Alphart Edu, use them as internal studio reasoning and then produce the final
game through `canvas_generate_game` or `generate_game`.

## Read These References

For a normal game generation request, load:

- `game-studio.md`
- `studio-design.md`
- `studio-dev.md`
- `studio-qa.md`

For larger requests, also load:

- `studio-brainstorm.md`
- `studio-plan.md`
- `studio-release.md`

## Alphart Edu Contract

- Do not call file-writing/coding tools such as `Write`, `Edit`, `MultiEdit`,
  `Bash`, `write_file`, `patch`, `terminal`, or `process`.
- Do not stop at a markdown plan.
- Do not call the game tool with only a prompt.
- Prefer a compact, complete, self-contained HTML document in the tool's `html`
  argument.
- Use `artifact_dir`, `artifact_path`, or `files` only when a real artifact
  exists and contains `index.html`.
- Never pass `files: []`.
- The generated game must be playable, accurate, classroom-safe, and bounded to
  one visible game stage that fits inside its iframe without page scrolling.
- Use an exact 1920x1080 logical stage. Scale that exact stage to the available
  viewport; do not create a larger scrollable page. A good baseline is:
  `html,body{margin:0;width:100%;height:100%;overflow:hidden}`
  `*{box-sizing:border-box}`
  `body{display:grid;place-items:center;background:#...}`
  `.stage{position:relative;width:1920px;height:1080px;overflow:hidden;transform-origin:top left}`
- The generated HTML itself must implement the scale-to-fit behavior, not rely
  on the Canvas iframe wrapper. Opening the public game URL in a normal browser
  tab must show the whole 1920x1080 stage without scrollbars.
- Keep HUD, buttons, dialogs, cards, sprites, labels, and tooltips inside the
  stage safe area. Do not use negative offsets, fixed overlays, oversized
  absolute panels, or transforms that move UI outside the stage.
- Do not use `position: fixed`; use absolute positioning inside the stage.

## Minimum QA Before Calling The Tool

- Visible game DOM exists directly in `<body>`.
- Start/restart controls are wired.
- Keyboard, mouse, touch, or button input mutates state.
- Score/progress/timer/level/lives/player position visibly changes through play.
- Win/fail/completion can be reached by playing.
- Text and widgets do not overflow or overlap.
- The game fits at 1920x1080, 1366x768, 1024x768, and 390x844 iframe sizes
  without clipped buttons, hidden text, horizontal/vertical page scroll, or
  overlapping panels.
- Every answer option group has a correct answer.
- Educational facts are precise and age-appropriate.
