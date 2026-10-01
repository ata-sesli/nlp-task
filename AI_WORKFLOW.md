# AI Workflow Summary

## Tools / Harnesses Used

- **Codex desktop** was my coding harness and conversational assistant for understanding the assignment, developing prompts, implementing the CLI, investigating failures, and reviewing results.
- **Antigravity CLI (`agy`)** was used for initial classification and extraction experiments. I replaced it with direct API requests because agent-driven file inspection, repeated crops, and inherited skills made execution slower and harder to control.
- The final Python CLI uses the **Google GenAI SDK**: `gemma-4-26b-a4b-it` with minimal thinking for classification, and `gemini-3.1-flash-lite` with high thinking for extraction. These inference requests have no agent tools or skills registered.
- **uv** manages the Python project; **Poppler** renders visible PDF pages, **Pillow** prepares images and views, and **jsonschema** validates structured outputs. Python performs reconciliation.

## Repository Instructions / Agent Configuration

- My global `~/.codex/AGENTS.md`, published as **[AGENTS.md](https://github.com/ata-sesli/AGENTS.md/blob/main/AGENTS.md)**, instructs Codex to inspect before changing code, keep scope narrow, make the smallest useful change, state uncertainty, verify APIs instead of inventing them, and show progress during long operations. There is no repository-local `AGENTS.md` in this submission.
- **[Octapowers skills](https://github.com/octacity-org/octapowers)** supported the development workflow, including security review of the earlier agy permission boundary. PDF skills supported visual checks. These are coding-assistant capabilities, not dependencies of the final extraction pipeline.
- **[Elephant MCP](https://github.com/octacity-org/elephant)** recalled project checkpoints and recorded implementation state between stages.
- **[codebase-memory-mcp](https://github.com/octacity-org/codebase-memory-mcp)** supported code discovery through an indexed knowledge graph. The instructions prefer graph searches and source snippets, with file searches when graph results are insufficient.
- The API key is loaded privately from the ignored `.env` file. I explicitly instructed the assistant not to expose its value in conversation, prompts, or logs.

## How I Used the Agent

- I divided the work into assignment interpretation, classification/grouping, extraction, reconciliation, CLI implementation, verification, and submission documentation. I discussed decisions with Codex before requesting scoped implementation changes.
- I chose the evidence priorities, required competing observations to remain in intermediate artifacts, and defined reconciliation as agreement across distinct documents rather than repeated page occurrences. I used Codex to draft prompts and schemas, implement the Python CLI, run experiments, and inspect saved outputs.
- Context included the assignment PDF, loan PDF, allowed labels, extraction-field definitions, output examples, and observed errors. I asked for read-only checks before changes and kept experimental runs separate from accepted artifacts.
- Classification receives the complete package. Extraction runs after classification, once per logical document group, retaining observations on their physical source pages. This choice preserves related context while reducing unrelated material in each request.
- I retained deterministic validation, formatting, normalization, document voting, and final publication in Python. After repeated service failures, I requested bounded HTTP 503 retries and explicit resume of compatible runs.

## Verification

- Codex reviewed code and diffs and added offline tests for page coverage, grouping, literal-value normalization, document voting, API configuration, retry limits, resume compatibility, timeout cleanup, secret redaction, and atomic publication. Mocked tests check implementation behavior; they do not establish model accuracy.
- Verification also included live API tests and visual comparisons with rendered PDF pages. These identified hidden-text leakage in earlier approaches, wrong field selection, wrong-page attribution, omissions, and altered quotations.
- The full run in `runs/full-grouped-sdk-20261001-213137` completed classification in about 64 seconds and grouped extraction in 128.7 seconds, including retries. Its final values and supporting pages matched the accepted results, and its `result.json` matched the root file byte for byte.
- **Known limitation:** this latest run omitted the visible competing affiant name `ALJA REGAN-VAN WILSEM` on page 3, although an earlier group test captured it. Some quotations flattened line breaks, and classification notes omitted the note pages' reversed printed order. These limitations do not change the selected final values, but I do not claim exhaustive extraction or a complete visual audit of every quotation. The latest raw output was not manually corrected.

## One Example Where the Agent Was Wrong or Incomplete

- **What happened:** the first full SDK run grouped the note pages correctly but returned `[7, 8, 10, 9]`, their logical reading order, despite the prompt requiring physical PDF order.
- **How I noticed:** Python validation rejected the group and stopped the pipeline before extraction. Comparing physical positions with printed page numbers explained the discrepancy.
- **What I changed:** I moved sorting into Python and simplified the prompt accordingly. Page entries and group page lists are sorted before validation, while duplicates remain detectable and the raw model response is preserved. An offline test covers this behavior. The subsequent classification run passed in 27 seconds with the expected labels and groups.
