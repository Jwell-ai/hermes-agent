---
name: game-studio
description: "Claude Code Game Studios workflow adapted for Alphart Edu playable educational HTML game generation."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, studio, education, html-game, pixel-art, qa]
    upstream: https://github.com/Donchitos/Claude-Code-Game-Studios
    source_commit_file: references/upstream/SOURCE_COMMIT
---

# Alphart Game Studio

This skill adapts the upstream
[`Donchitos/Claude-Code-Game-Studios`](https://github.com/Donchitos/Claude-Code-Game-Studios)
workflow for Alphart Edu.

Use this skill when the user asks for a playable educational game, interactive
demo, quiz game, simulation, platformer, boss challenge, story quest, or similar
activity.

## Load Upstream Guidance

For non-trivial game requests, read these references before calling the game tool:

- `references/upstream/README.md`
- `references/upstream/docs/quick-start.md`
- `references/upstream/docs/agent-roster.md`
- `references/upstream/docs/workflow-catalog.yaml`
- `references/upstream/docs/director-gates.md`
- `references/upstream/docs/review-workflow.md`
- `references/upstream/docs/coordination-rules.md`

For implementation and QA constraints, also read:

- `references/upstream/rules/gameplay-code.md`
- `references/upstream/rules/ui-code.md`
- `references/upstream/rules/prototype-code.md`
- `references/upstream/rules/test-standards.md`

Use upstream studio roles as reasoning lenses, not as external processes. In this
agent service there are no Claude Code slash commands or file-writing tools.

## Alphart Edu Contract

The final artifact must be uploaded through `canvas_generate_game` or
`generate_game`.

- Do not call `Write`, `Edit`, `MultiEdit`, `Bash`, `write_file`, `patch`,
  `terminal`, or any local file/process tool for game requests.
- Do not stop at a markdown plan.
- Do not call the game tool with only a prompt.
- Prefer passing one compact, complete HTML document in the tool's `html`
  argument.
- Use `artifact_dir`, `artifact_path`, or `files` only when a real generated
  artifact directory/file list exists and contains `index.html`.
- Never pass `files: []`.

## Studio Workflow For Alphart

1. **Creative Director / Game Designer**
   - Define the player fantasy, learning goal, target audience, and game pattern.
   - Choose a real playable pattern: pixel platformer, top-down maze, arcade
     matcher, drag sorter, simulation sandbox, boss challenge, physics launcher,
     or story quest.
   - Use quiz/card/form only when the user explicitly asks for a quiz.

2. **Systems Designer**
   - Create a `content_facts` ledger.
   - Every question, label, hazard, collectible, feedback message, and answer
     option must trace to that ledger, the user request, or stable common
     knowledge.
   - Every answer-option group must have at least one correct choice.
   - Single-answer questions must have exactly one correct choice.
   - Multi-select questions must clearly say multi-select and have one or more
     correct choices.

3. **Prototyper / UI Designer**
   - Build a complete playable HTML game.
   - Default visual style is simple pixel art: crisp edges, tiled playfield, HUD,
     visible player/targets/hazards, limited high-contrast palette, classroom-safe
     colors.

4. **QA Lead**
   - Mentally playtest the game before calling the tool.
   - Reject static explanation pages, fake progress, unwired buttons, placeholder
     TODOs, impossible questions, unplayable controls, clipped text, overlap, or
     overflow.
   - Verify content is accurate, age-appropriate, non-vulgar, non-sexual,
     non-hateful, non-graphic, and classroom safe.

5. **Upload**
   - Call `canvas_generate_game` or `generate_game` only after the HTML passes QA.
   - If validation fails, correct the HTML once using the error text as a hard
     requirement, then call the tool again.

## HTML Requirements

The generated HTML must:

- Start with `<!DOCTYPE html>` and end with `</html>`.
- Include visible first-paint game DOM directly in `<body>`, not only elements
  created later by JavaScript.
- Include `<main id="game-root">` directly in `<body>`.
- Use a fixed 1920x1080 logical game stage.
- Center and scale the stage for smaller viewport sizes.
- Use `overflow:hidden` on `html`, `body`, and the root stage.
- Avoid `position: fixed`, negative offsets, viewport-sized overlays, and
  transforms that push UI outside the stage.
- Keep all UI inside the safe area: x=40..1880 and y=40..1040.
- Include visible HUD/score/progress, playfield/canvas/SVG, instructions, and
  start/restart controls.
- Wire keyboard, mouse, touch, or button input to mutate state.
- Include a real update/render loop or event-driven game state changes.
- Make score/progress/timer/level/lives/player position visibly change through
  play.
- Make win/fail/completion reachable by playing.

Minimum body shape:

```html
<body>
  <main id="game-root">
    <section id="hud">Score / progress / goal</section>
    <section id="playfield">
      <canvas id="game-canvas" width="1600" height="760"></canvas>
    </section>
    <section id="instructions">Goal and controls</section>
    <button id="start-btn">Start</button>
    <button id="restart-btn">Restart</button>
  </main>
  <script>
    // Wire controls and game loop here.
  </script>
</body>
```

## Content Precision

Educational accuracy is more important than novelty.

- Do not invent formulas, dates, names, units, definitions, or causal claims.
- Use only the user request and stable common knowledge.
- If the source topic is ambiguous, keep claims general and safe.
- Feedback must explain why an action or answer is correct or wrong.
- Hazards should represent misconceptions or abstract obstacles, not graphic
  violence.
