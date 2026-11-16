# Contribute to dl-lowlat-infer

## Development workflow

1. Fork [NVIDIA/dl-lowlat-infer](https://github.com/NVIDIA/dl-lowlat-infer), or
   use a branch in the main repository if you have write access.
2. Create a feature or bug-fix branch from `main`.
3. Make sure pre-commit checks pass and every commit is signed off.
4. Open a pull request against `main` and describe the change and how you tested it.
5. A maintainer reviews and merges the pull request.

Do not report security vulnerabilities through public pull requests or issues.
Follow [SECURITY.md](SECURITY.md) instead.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) in this form:

```text
<type>(optional-scope): <imperative summary>
```

Keep the summary lowercase and do not end it with a period.

Use one of these types:

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

Examples:

- `perf(single-step): speed up the precompute`
- `fix(gdrcopy): order the signal write on aarch64`
- `docs: document the contribution workflow`

## Pre-commit checks

The complete formatting and static-analysis workflow is documented in
[CODING_GUIDELINES.md](CODING_GUIDELINES.md).

Install and enable the hooks once per clone:

```shell
pip install pre-commit
pre-commit install
```

The hooks run on every commit. They perform repository hygiene, formatting,
linting, spelling, and DCO checks. Run every check manually with:

```shell
pre-commit run --all-files
```

## Sign your work

All contributors must sign off every commit as described below.

The sign-off is a simple line at the end of the explanation for the patch. Your
signature certifies that you wrote the patch or otherwise have the right to pass
it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Your Name <your@email.com>

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.
