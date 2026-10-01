# loancli

A small Python CLI that classifies, extracts, and reconciles the provided
12-page loan package using the official `google-genai` SDK.

## Setup and run

Install [uv](https://docs.astral.sh/uv/) and Poppler (`pdftoppm` must be on
`PATH`). Run from this project directory:

```sh
uv sync
```

Create a local `.env` containing your key:

```dotenv
GEMINI_API_KEY=your-key-here
```

`.env` is ignored by Git. The worker reads the key privately from this
file, without changing environment variables or saving it in prompts,
request artifacts, or logs. An ambient API key is not used.

```sh
uv run loancli classify --output runs/classify

uv run loancli extract \
  --classification runs/classify/classification.json \
  --output runs/extract

uv run loancli full --output runs/full
```

`--classification` accepts clean classification JSON or a successful saved
run envelope, including older agy envelopes. Use a new or empty output
directory for a fresh run. To explicitly resume an interrupted run:

```sh
uv run loancli full --output runs/full --resume

uv run loancli extract \
  --classification runs/classify/classification.json \
  --output runs/extract --resume
```

Resume requires the same command, PDF, prompts, schemas, classification
input (for `extract`), and model/runner configuration as recorded in
`run-manifest.json`. Completed classification and extraction groups are
revalidated against their raw responses before reuse. Changed inputs or
missing manifests fail before any model request. Older runs without a
manifest cannot be resumed. The timeout may be changed on resume; each
invocation receives a fresh stage deadline. API keys are excluded from
the manifest.

Classification makes one `generate_content` request to `gemma-4-26b-a4b-it`
with minimal thinking. Extraction makes one request per logical document
group from classification, using `gemini-3.1-flash-lite` with high thinking.
Both use JSON response MIME type and API-configured `response_json_schema`.
No tools are registered.
HTTP 503 failures receive at most three retries per request, after 5, 10,
and 20 seconds (four attempts total), using the same model and inputs.
Each attempt and retry delay is shown in progress and saved in the JSONL
log. Other errors, including unsupported configuration, fail immediately.
There are no fallback models. The hard timeout can stop retries before
the limit. Gemma's
schema support was confirmed by the project owner; Google's
[Flash-Lite documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
confirms structured outputs and high thinking. Offline tests do not
verify live service capabilities.

## Outputs and timeouts

| Command | Structured artifacts |
| --- | --- |
| `classify` | `classification.json` with page labels and document groups |
| `extract` | Supplied `classification.json` and new `extraction.json` observations |
| `full` | Both stages, `reconciliation.json`, and final `result.json` |

After successful reconciliation, `full` atomically replaces the project
root's `result.json` with the same bytes saved under `--output`. Failures
leave the existing root result unchanged. `classify` and `extract` do not
update it.

Classification saves prompt and schema snapshots, `classification-run.json`,
`classification.stream.jsonl`, `classification.stderr.log`, and
`classification.render.log`. Each extraction group saves the corresponding
`extraction.*` files and raw `extraction-run.json` under
`groups/document-01/`, `groups/document-02/`, and so on. Group schemas
restrict page numbers and entry counts to that group's physical pages.

After every group succeeds, Python merges observations into
`extraction.json` in physical page order using the existing full-package
schema. The top-level `extraction-run.json` records the successful group
artifacts and merged output. A failed group stops processing; completed
groups and the failing group's logs remain, and no merged extraction or
final result is published. Explicit resume reuses completed groups and
starts the failed group's request again with its bounded retry allowance.
Only complete extraction can reach reconciliation and final publication.

The JSONL log contains request lifecycle events, returned response text,
and the final result; it is not a token stream. Hidden thinking is not
saved. Progress, heartbeats, and stage durations appear on stderr.
Logs are appended on resume, with invocation markers distinguishing
attempts from different invocations; earlier response text and errors remain.

All commands accept `--timeout SECONDS` (default: 300), separately for each
stage, including image preparation and the model calls. The extraction
stage shares this deadline across all groups; it does not reset per group.
Python runs the SDK call in a separate process. Timeout or Ctrl-C terminates its process
group and remaining children, preserves logs, and cleans temporary files.
Cleanup can take up to five additional seconds. Failures return a nonzero
exit code and retain completed artifacts.

## Processing rules

Prompts are read from `prompts/`; schemas from `schemas/` configure the API
response and validate it locally. Python renders the PDF at 300 DPI. In
`full`, it renders once in a shared disposable directory and reuses those
images for both stages. Standalone commands render their own images.
Each stage uses a separate `.temp/<run-id>/<stage>/` directory with regular
image copies. Shared and stage preparation files are removed on success
or failure.

Classification receives all 12 full-page images together as Pillow objects,
preceded by physical page-number labels. Each extraction request receives
only its group's full-page images, classification labels and group context,
and three prepared views per page: top 0–40%, middle 30–70%, and bottom
60–100%.
Pillow crops these overlapping bands from the same rendered images,
without resampling or rendering PDF regions again. Extraction gets inline
classification labels and groups, excluding classification notes and
previous extraction results. No agy settings, skills, or conversations are
loaded; global coding configuration remains unchanged.

Both prompts require direct reading of visible evidence. Unreadable text
remains valid intermediate uncertainty, with null values and explanatory
notes. Structural checks enforce complete physical page coverage, group
consistency, field statuses, and supporting evidence. They cannot prove
that a quotation matches the image; evaluating model accuracy still
requires visual verification. Whole-package extraction accuracy is not
established by the preliminary group tests.

Reconciliation normalizes whitespace, capitalization, unambiguous
surname–given-name ordering, and address separator commas. It preserves
spelling, digits, leading zeros, hyphens, and internal word-spacing
differences. Each logical document contributes at most one vote per
candidate; a document may support competing candidates once each.
Original observations, alternatives, source pages, and selection rationale
remain in intermediate artifacts.

The unique candidate with the most document votes wins. An exact tie is
saved in `reconciliation.json` and fails the run without a final result:
a tie selection rule remains undecided. If a field has no readable
candidate, its final value and source pages are empty, with uncertainty
retained in reconciliation.

## Offline tests

```sh
uv run --offline python -m unittest discover -s tests -v
```

Tests mock the SDK client and use accepted saved results. They cover all
three commands, one extraction request per group, single rendering in full mode,
fixed views, group coverage and failure preservation, merging, document
voting, ties, bounded 503 retries, resume compatibility checks, secret redaction, timeout
cleanup, and atomic publication. Most tests use a fake renderer; one uses
real Poppler. No model calls are made.
