# loancli

A small Python CLI for classifying, extracting, and reconciling the
provided 12-page loan package with Antigravity.

## Run

Run from this project directory. Install [uv](https://docs.astral.sh/uv/)
and authenticate `agy` first. `agy` must be on your `PATH`; the agent
Python uses Poppler's `pdftoppm` to render the PDF before each model call;
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

The CLI reuses `classification.prompt.txt`, `extraction.prompt.txt`, and
their schemas. Each call creates a unique `.temp/<run-id>/<stage>/`
working directory. Python freshly renders the PDF and copies only the 12
page images, schema, effective prompt, and required stage inputs into it,
using regular copies rather than symlinks. The effective prompt directs
the model to inspect those images instead of finding or rendering a PDF.
Extraction receives classification labels and groups; classification
notes and former extraction results are excluded. The entire temporary
run directory is removed on success or failure; logs and results stay
under `--output`.

Each call launches a fresh `agy` process with `--new-project --sandbox
--disable-slash-commands`. It does not resume conversations or add
workspace directories. The streamed `init.cwd` must match the stage
directory before the output is accepted. Global `agy` settings are left
unchanged. Inherited settings may permit outside access: this is a clean
workspace, not guaranteed filesystem isolation.

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
three commands without launching a model.
