---
name: studio-qa
description: "Game Studio QA phase — QA Lead writes test plans and regression suites; testers execute; bugs are triaged and tracked."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [game, qa, testing, regression, bug-triage, qa-lead, playtest]
    related_skills: [game-studio, studio-dev, studio-release]
---

# Studio · QA Phase

**Active roles:** QA Lead (plan + triage) · QA Tester (execution) · Lead Programmer (bug fixes)

## Goal
Produce a QA plan for a feature or area, generate test cases, execute them against the
described behaviour, triage any failures, and produce a signed-off QA report.

## Invocation
```
/studio qa <area>
# examples:
/studio qa combat system
/studio qa save/load
/studio qa full regression
/studio qa playtest report
```

## Process

### Step 1 — Scope
Identify what is under test:
- Feature name and GDD reference
- Stories in scope (from backlog)
- Engine and platform

### Step 2 — Test Plan
```markdown
## QA Plan: <area> — <date>

### Scope
…

### Test Environment
Engine: … | Platform: … | Build: …

### Risk Areas (highest-priority testing)
1. …
2. …

### Out of Scope
…
```

### Step 3 — Test Cases
For each risk area, write test cases:
```
TC-[n]: <title>
Precondition : …
Steps        : 1. … 2. … 3. …
Expected     : …
Actual       : [PASS / FAIL / BLOCKED — fill in during execution]
Severity     : Critical · High · Medium · Low
```

### Step 4 — Bug Triage
For each FAIL:
```
BUG-[n]: <title>
Repro Steps : …
Expected    : …
Actual      : …
Severity    : Critical · High · Medium · Low
Assigned To : Lead Programmer / Designer / Artist
```

Critical bugs block release. High bugs block story done. Medium/Low go to backlog.

### Step 5 — QA Report
```markdown
## QA Report: <area>

| Total TCs | Pass | Fail | Blocked |
|-----------|------|------|---------|
| …         | …    | …    | …       |

Critical bugs open: [n] — **RELEASE BLOCKED** (if n > 0)
High bugs open   : [n]
Sign-off         : QA Lead — [APPROVED / BLOCKED]
```

Write report to `production/qa-report-<area-slug>.md`.
Offer to continue with `/studio release` when all critical bugs are resolved.
