<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# AGENTS.md

## Project

This is a Dagster project. Assets, jobs, schedules and sensors go in `src/cv_lakehouse/defs/`.
Dagster loads that folder automatically. Tests go in `tests/`.

## Bronze

Bronze holds a dataset exactly as its publisher distributes it. Many use cases read it,
so bronze decides nothing about the content.

- Download every file that the publisher publishes. Skip nothing, such as a split
  without labels or an evaluation kit.
- Keep the bytes. Do not rewrite, filter, re-encode or rename a file.
- Reproduce a manual download: each file at its published name, each archive unpacked
  beside itself.
- Keep every archive after it is unpacked.
- Pin every file to a size and a checksum, and verify it.
- A linked copy must hold every published part, or the link fails.
- Put every decision about the content in silver.

A source lists its files in `published_files`. It holds no download code.

## Commands

Use the Makefile. Do not call `uv run`, `ruff`, `pyrefly` or `pytest` directly.

| Command | Action |
| --- | --- |
| `make sync` | Install dependencies from the lock file. |
| `make format` | Format the code, and add the licence headers. |
| `make lint` | Report lint problems. |
| `make lint-fix` | Fix lint problems. |
| `make typecheck` | Check the types. |
| `make test` | Run the unit tests. |
| `make defs` | Validate the Dagster definitions. |
| `make check` | Run every check. |
| `make dev` | Start the Dagster UI on port 3000. |
| `make image` | Build the Dagster code location image. |

Run `make check` before you finish a task.

## Commits

Write every commit as a Conventional Commit. release-please reads the commits to pick
the next version and to write the changelog.

- Squash merge every pull request. The PR title becomes the commit, so the title is
  what counts.
- Use one of these types: `feat`, `fix`, `perf`, `refactor`, `docs`, `build`, `ci`,
  `test`, `style`, `chore` or `revert`.
- Start the subject in lower case, and end it without a period.
- For a breaking change, write `!` after the type, or a `BREAKING CHANGE:` footer.

## Releases

The app and `triton/` release independently. A commit under `triton/` goes into the next
`triton` release. Any other commit goes into the next app release.

- release-please keeps one release PR open for each of them. Merging it tags the
  release and publishes the image.
- Do not edit a version, a changelog or `.release-please-manifest.json` by hand.

## Licence headers

Every source file starts with the MIT header in `dev/licenseheader_lionelpeer.tmpl`.
`make format` adds it, and `make check` fails on a file without it. An `__init__.py`
stays empty and gets no header.

Leave a blank line between the header and a comment under it. Otherwise `make format`
takes the comment for a part of the header, and deletes it.

## Dependencies

To add a dependency, run `uv add <package>`. To add a development dependency, run
`uv add --dev <package>`. Both commands update `uv.lock`. Commit `uv.lock`.

Every Makefile target uses `--frozen`. If the lock file is out of date, the targets fail.
Run `make lock` to update it.

## Packages

Every `__init__.py` file stays empty. A module that other code imports lives in a named
file, not in the package root. Import the file, not the package.

## Protocols

A class that satisfies a Protocol inherits it. The type checker then verifies the class
where it is written, not only where it is used.

## Self

A method that returns an instance of its own class annotates the return type `Self`. A
subclass then gets its own type back. A constructor that needs `Self` is a `classmethod`,
not a `staticmethod`, and it builds the instance with `cls`.

## Calls

Pass every argument by keyword. A call that passes one argument passes it positionally.

```python
write_manifest(path=silver_dir / SILVER_MANIFEST, manifest=manifest)
read_image_size(path)
```

A class, a dataclass and a pydantic model are called the same way.

Some callees take no keyword. Pass those arguments positionally:

- A parameter that is positional-only, such as in `min`, `isinstance`, `zip`,
  `dict.get`, `dataclasses.replace` and `np.concatenate`.
- A function that collects `*args`, such as `parser.add_argument` and
  `ThreadPoolExecutor.map`.

Do not write `*args` in a new function. Take one sequence, so a caller can name it.

Where a new function takes three or more parameters, write `*` before them. The
signature then rejects a positional call, as `write_split` and `_embed_split` do.

## Naming

A name describes what the thing is. A function name describes what the function does.

Write a name that makes a docstring unnecessary. If a docstring repeats the name, delete
the docstring.

- Start a function name with a verb. Write `detect_materialized_splits`, not
  `materialized_splits`.
- An accessor and an alternative constructor name the value instead. Write
  `bronze_dir`, and write `Crop.rounded_to_pixels`.
- Qualify a name that fits any object. Write `class_registry`, not `registry`.
- Name a function for its action, not for the property it tests. Write
  `reject_repeated_names`, not `unique`.
- Give one name one meaning in the whole package. Do not reuse a function name for a
  property or a variable.

## Comments and documentation

Write the absolute minimum. Prefer clear code over a comment.

Delete a comment that repeats the code. Write a comment only for a decision that the code
cannot show, such as a workaround or a non-obvious constraint.

Follow SimpleEnglish (https://github.com/AminBlg/SimpleEnglish) for all prose:

- Maximum 20 words per instruction. Maximum 25 words per description.
- Put the condition before the command.
- Use simple tenses and the active voice.
- Do not use should, would, may or might.
- One word has one meaning in the whole document.
- Keep the articles. Keep the word that.
- Do not use a bold lead-in. Do not write a heading of more than two sentences.
- State the fact, not its importance.
- Do not use an em-dash.
