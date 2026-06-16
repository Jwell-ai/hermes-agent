---
name: studio-release
description: "Game Studio release phase — Release Manager runs the launch checklist, coordinates day-one patch, and signs off for deployment."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, release, launch, checklist, day-one-patch, release-manager, ship]
    related_skills: [game-studio, studio-qa, studio-plan]
---

# Studio · Release Phase

**Active roles:** Release Manager (gate authority) · Producer (scope sign-off) · QA Lead (final regression sign-off)

## Goal
Execute the full release checklist, identify any blocking items, prepare a day-one patch
plan if needed, and produce a signed release artefact.

## Invocation
```
/studio release
/studio release <version-tag>
```

## Pre-Release Gate

All items must be ✅ before proceeding:

### Code & Build
- [ ] All P1 stories marked DONE
- [ ] No open Critical or High bugs (QA report sign-off exists)
- [ ] Build reproducible from clean checkout (`git clean -fdx && build`)
- [ ] No hardcoded dev/test credentials in source
- [ ] Version constant updated in code and project settings

### Content & Assets
- [ ] All placeholder assets replaced
- [ ] Localisation strings complete for launch languages
- [ ] Audio mix final (no temp SFX or music)
- [ ] All UI strings reviewed for spelling/grammar

### Platform & Distribution
- [ ] Build tested on each target platform
- [ ] Store page / distribution metadata complete
- [ ] ESRB / PEGI / age-rating submitted (if applicable)
- [ ] Analytics events firing correctly on key interactions

### Legal & Compliance
- [ ] Third-party licences documented in `CREDITS.md`
- [ ] No GPL-incompatible assets in commercial build
- [ ] Privacy policy URL set in store metadata

## Day-One Patch Plan
If any High bugs remain open, document them here:
```
KNOWN ISSUE [n]: <title>
Impact    : …
Workaround: …
Fix ETA   : Day-one patch / Patch 1.0.1 / v1.1
```

## Release Sign-Off Document
Write to `production/release-<version>.md`:
```markdown
# Release Sign-Off: <version> — <date>

## Gate Status
[checklist results]

## Open Known Issues
[day-one patch plan]

## Approvals
- QA Lead      : [APPROVED / BLOCKED]
- Producer     : [APPROVED / BLOCKED]
- Release Manager: [APPROVED / BLOCKED]

## Deployment Notes
…
```

If all approvals are APPROVED, print:
```
🚀 RELEASE APPROVED: <version>
Artefact: production/release-<version>.md
```

Otherwise print the blocking items and recommend fixes.
