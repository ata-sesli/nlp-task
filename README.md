# loancli

A small Python CLI for classifying, extracting, and reconciling the
provided 12-page loan package with Antigravity.

## Run

Run from this project directory. Install [uv](https://docs.astral.sh/uv/)
and authenticate `agy` first. `agy` must be on your `PATH`.
Python uses Poppler's `pdftoppm` to render the PDF before the model calls;
it must also be on your `PATH`. `uv run` installs the local CLI and its one runtime dependency,
`jsonschema`.

```sh
uv run loancli classify --output runs/classify

uv run loancli extract \
  --classification runs/classify/classification.json \
  --output runs/extract

uv run loancli full --output runs/full
```

`--classification` accepts clean classification JSON or a successful
saved `agy` JSON envelope. Outputs must go into a new or empty directory;
use a different directory for each rerun.

All commands accept `--timeout SECONDS` (default: 300). The hard limit
applies separately to each stage, including image preparation and the
model call. For example, `uv run loancli full --output runs/full --timeout 180`
allows up to 180 seconds per stage. On timeout or interruption, the
runner terminates the process group, kills remaining children, cleans
the temporary workspace, and preserves logs. Cleanup may take up to
five additional seconds.

## What each command saves

| Command | Structured artifacts |
| --- | --- |
| `classify` | `classification.json` with page labels and document groups |
| `extract` | Supplied `classification.json` and new `extraction.json` observations |
| `full` | Both stages, `reconciliation.json`, and final `result.json` |

After successful reconciliation, `full` also atomically replaces the
project root's `result.json` with the same bytes saved under `--output`.
Failures leave the existing root result unchanged. `classify` and
`extract` do not update it.

Each model stage also saves its prompt and schema snapshots, raw
`<stage>.stream.jsonl`, `<stage>.stderr.log`, `<stage>.render.log`, and terminal
`<stage>-run.json`. Progress appears on stderr, including a heartbeat
while waiting. Failures return a nonzero exit code and preserve the logs
and completed artifacts. Ctrl-C stops the runner and preserves partial logs.

## Processing rules

The CLI reads `classification.prompt.txt` and `extraction.prompt.txt`
from `prompts/`, and their schemas from `schemas/` for validation and
stage inputs. Each call creates a unique `.temp/<run-id>/<stage>/`
working directory. In `full`, Python renders the PDF once at 300 DPI in
a shared disposable preparation directory and copies the same images
into each stage's separate workspace. Standalone commands render their
own images. Python copies
the 12 page images, schema, effective prompt, and required stage inputs into it,
using regular copies rather than symlinks. The effective prompt directs
the model to inspect those images instead of finding or rendering a PDF.
Extraction receives classification labels and groups; classification
notes and former extraction results are excluded. Python also prepares
three full-width views from the rendered image pixels, without rendering
PDF regions again: top 0–40%, middle 30–70%,
and bottom 60–100%. These overlapping views retain 300 DPI source detail
without resampling and let the model inspect unclear text at a larger
display size. Their fixed names are `page_views/page-01-top.png`,
`page-01-middle.png`, and `page-01-bottom.png`, through page 12.
The model uses these supplied views without creating crops or running
image-processing commands. If text remains unreadable, null values and
uncertainty notes remain valid intermediate observations and receive
no document votes. Stage workspaces and shared preparation files
are removed on success or failure; logs and results stay
under `--output`.

Each call launches a fresh `agy` process with `--new-project --sandbox
--disable-slash-commands`. It does not resume conversations or add
workspace directories. The streamed `init.cwd` must match the stage
directory before the output is accepted. Global `agy` settings are left
unchanged. Inherited settings may permit outside access: this is a clean
workspace, not guaranteed filesystem isolation.

On macOS, loancli additionally launches agy inside a per-process
`sandbox-exec` profile that blocks reads of `~/.gemini/config/skills/`,
`~/.agents/skills/`, and the Octapowers router in `~/.gemini/GEMINI.md`.
This keeps global coding skills out of loan processing without editing
or renaming shared files; ordinary coding sessions retain their setup.
Other inherited settings remain active. This isolation currently requires
macOS; `sandbox-exec` is deprecated by Apple but remains available and is
also used by agy's macOS terminal sandbox.

Every run uses `gemini-3.8-flash-low` and
[`agy --output-format stream-json`](https://antigravity.google/docs/cli/headless/).
Only the terminal `result.structured_output` is accepted. Schema checks,
complete physical page coverage, group consistency, field statuses, and
required supporting evidence are validated before publishing clean outputs.
These checks cannot prove that a quotation matches the rendered image;
visual verification remains necessary when evaluating model accuracy.

Reconciliation normalizes whitespace, capitalization, unambiguous
surname–given-name ordering, and address separator commas. It preserves
spelling, digits, leading zeros, hyphens, and internal word-spacing
differences. Each logical document contributes at most one vote per
candidate; a document may support competing candidates once each.
All original observations, alternatives, source pages, and selection
rationale remain in the intermediate artifacts.

The unique candidate with the most document votes wins. An exact tie
is recorded in `reconciliation.json` and reported as a failure; no final
`result.json` is written because a tie selection rule is still undecided.
If a field has no readable candidate, its final value is empty and its
source pages are empty, with the uncertainty retained in reconciliation.

## Test without model calls

```sh
uv run python -m unittest discover -s tests -v
```

Tests use the accepted saved results and a fake `agy` executable. They
exercise validation, streaming failures, document voting, ties, and all
three commands, including timeout cleanup and fixed views, without
launching a model. Most tests also use a fake renderer; one uses real Poppler.
