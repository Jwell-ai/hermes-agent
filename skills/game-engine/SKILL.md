---
name: game-engine
description: "Build complete 2D educational web games with HTML5 Canvas and JavaScript."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, 2d, canvas, html5, javascript, education, pixel-art]
    upstream: https://github.com/github/awesome-copilot/tree/main/skills/game-engine
---

# 2D Game Engine

Use this skill for playable 2D educational games: platformers, mazes, arcade
matchers, drag sorters, simulations, physics launchers, and story quests.

This is an Alphart Edu adaptation of GitHub's `game-engine` skill. Use HTML5
Canvas and plain JavaScript. Do not create 3D, WebGL, multiplayer, or framework
dependent games.

## Core Loop

Every game needs a playable loop:

1. Read keyboard, mouse, touch, or button input.
2. Update state, movement, collision, score, timer, or selection.
3. Render the changed game state with `requestAnimationFrame` or a clear
   event-driven equivalent.
4. Give immediate feedback and make a win, loss, or completion state reachable.

Use Canvas 2D for the playfield and DOM only for the HUD, controls, and
accessible instructions. Use simple AABB or circle collision detection when
the game pattern needs it. Use delta time for continuous movement.

## Educational Game Design

- Start by defining the learning goal, target audience, core mechanic,
  controls, win state, and the precise facts used in the game.
- Default to simple pixel art: crisp edges, tiled playfields, visible player,
  targets, HUD, and a limited high-contrast palette.
- Use a quiz/card form only when the user explicitly requests a quiz. Prefer
  an interaction where learning affects play: collect correct facts, sort
  concepts, navigate a maze, match terms, or simulate a system.
- Every answer set needs at least one correct answer. Single-answer questions
  need exactly one correct answer; multi-select questions must say so.
- Do not invent formulas, dates, units, definitions, names, or causal claims.
  Keep content accurate, age appropriate, and classroom safe.
- Use non-graphic metaphors for hazards and failure, such as obstacles,
  misconception blockers, puzzles, or energy loss.

## Alphart Upload Contract

The final artifact must be sent through `canvas_generate_game` or
`generate_game`.

- Do not use file-writing, shell, patch, or terminal tools.
- Do not stop at a plan and do not call the game tool with only a prompt.
- Pass a compact, complete, self-contained HTML document through `html`.
- Do not depend on external CDNs, third-party game libraries, remote assets,
  build steps, or a local development server.
- The document must begin with `<!DOCTYPE html>` and finish with `</html>`.
- Include visible first-paint game DOM directly in `<body>`: a semantic game
  root such as `<main>`, a HUD, a Canvas/SVG playfield, instructions, and real
  start/restart controls. The root id is your choice.

## Stage And Layout

Generated games must use a fixed 1920x1080 logical stage that scales to fit
the actual iframe or browser tab. The public game URL must show the whole game
without scrollbars.

- Set `html`, `body`, and the game stage to `overflow:hidden`.
- Center and scale the 1920x1080 stage with JavaScript using
  `min(innerWidth / 1920, innerHeight / 1080)`.
- Keep all controls, HUD panels, dialogs, sprites, labels, and tooltips inside
  the stage safe area: x=40..1880, y=40..1040.
- Do not use `position: fixed`, negative offsets, oversized viewport panels,
  or transforms that move content outside the stage.
- Include keyboard and touch/click controls where appropriate. A game must be
  playable on both desktop and touch-sized viewports.
- For reliable runtime validation, expose `window.__ALPHART_GAME_TEST__` with a
  `start()` function when practical. It should start the game and visibly
  change the game state; otherwise the harness invokes the first enabled
  start/restart button.

## Preflight Check

Before calling the upload tool, verify that:

- the initial HTML visibly renders a game shell;
- start and restart controls have real handlers;
- input changes visible state;
- score, progress, timer, level, lives, or player position changes during play;
- collisions, answer checking, or target validation is implemented;
- the player can reach completion;
- there are no placeholder controls, clipped labels, overlap, or page scroll;
- the game fits at 1920x1080, 1366x768, 1024x768, and narrow mobile viewports.

The upload tool runs the same artifact harness. It rejects incomplete HTML,
external CDN/media dependencies, missing playfields or controls, absent
interaction/UI-update signals, stage escape CSS, missing 1920x1080 scale
logic, and TODO/placeholder artifacts. When Browserless is configured, it also
renders the final game at stage, laptop, and mobile viewports and rejects
runtime errors, missing visible controls/playfields, offscreen stages, and
document overflow. It serves any local artifact assets from an isolated
in-memory origin, invokes a game start control, and requires a visible state
change. Treat a harness error as a hard requirement, correct the
HTML, and retry once.

If the upload tool reports a game validation error, regenerate the HTML once
with the returned error as a hard requirement and retry once.
