# Coding Guidelines

## Pre-commit

Install and enable the repository checks once per clone:

```shell
pip install pre-commit
pre-commit install
```

The checks then run on every commit. Run them manually across the repository with:

```shell
pre-commit run --all-files
```

The configured hooks reject:

- added large files
- broken symlinks
- merge-conflict markers
- private keys
- malformed YAML, JSON, or TOML
- mixed line endings
- debugging statements
- spelling errors
- trailing whitespace
- missing final newlines

They also format C/C++/CUDA, CMake, and Python files. The commit-message hook
requires the DCO sign-off added by `git commit -s`.

## C++ formatting

C++, C, and CUDA files are formatted with `clang-format` using the repository
`.clang-format` file.

## C++ static analysis

`clang-tidy` runs in merge-request CI across all registered host C++
translation units. It is kept out of pre-commit because it needs a configured
build and is substantially slower than the formatting hooks.

All enabled checks report diagnostics. The `.clang-tidy` configuration treats
the following diagnostic families as errors:

- `clang-diagnostic-*`,
- `clang-analyzer-*`,
- `bugprone-*`,
- `performance-*`, and
- `portability-*`.

Other check families remain informational.

Generate the compilation database and run a complete local host C++ scan with:

```shell
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCPU_ARCH=native \
    -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_TESTS=OFF \
    -B build
python3 scripts/run_clang_tidy.py --build-dir build
```

The clang-tidy integration analyzes all first-party `.cpp`, `.cc`, and `.cxx`
translation units in the compilation database. CUDA translation units continue
to be checked by `clang-format`; clang-tidy coverage for NVCC compilation
commands will be enabled separately after the invocation is validated.

## CMake formatting

CMake files are formatted with `cmake-format` using `.cmake-format.json`.

## Python formatting

Python code is checked and formatted with Ruff:

```shell
ruff check --fix src/python test/python scripts
ruff format src/python test/python scripts
```

Ruff enforces pycodestyle, Pyflakes, import ordering, selected pylint errors,
and Google-style docstrings. Existing modules are not required to add missing
module, class, function, or method docstrings.

## Spelling

Codespell checks source code and documentation. Add genuine project terminology
to the repository allowlist instead of suppressing spelling checks for a whole
file.
