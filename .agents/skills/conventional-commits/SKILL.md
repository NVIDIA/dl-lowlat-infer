---
name: conventional-commits
description: >-
  Conventional Commit message format for this repo: commit types, imperative
  summaries, scopes, squash-merge bodies, and DCO sign-off. Use when writing,
  reviewing, or amending git commit messages.
---

# Conventional Commits

Every commit message uses:

```text
<type>(optional-scope): <imperative summary>
```

## Types

| Type | Usage |
| --- | --- |
| `feat` | New functionality |
| `fix` | Bug fixes |
| `perf` | Performance improvements |
| `refactor` | Behavior-preserving code changes |
| `docs` | Documentation-only changes |
| `test` | Test additions or corrections |
| `build` | Build-system or dependency changes |
| `ci` | Continuous-integration changes |
| `chore` | Maintenance not covered above |
| `revert` | Revert an earlier commit |

## Rules

- Use the imperative mood: "add feature", not "added feature" or "adds feature".
- Keep the summary lowercase and concise; no trailing period.
- Add a scope when it aids scanning (component or area, e.g. `python`, `cpp`).
- Release squash commits must list every included change in the commit body.

## Examples

```text
perf(single-step): speed up the precompute
fix(gdrcopy): order the signal write on aarch64
docs: document the contribution workflow
```

## DCO sign-off (required)

Every commit must end with a sign-off line certifying the Developer
Certificate of Origin (developercertificate.org):

```text
Signed-off-by: Your Name <your@email.com>
```

Sign automatically from configured `user.name`/`user.email` with `git commit -s`.
