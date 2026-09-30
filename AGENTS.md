# AGENTS.md

This file is the canonical implementation guide for all coding agents working
in this repository. It is provider-agnostic: the rules apply equally to
Claude, Codex, and other automated coding tools. Read it in full before making
changes.

## Project overview

SleuthBench is an automated QA-generation system for benchmarking LLMs on
tabular data analysis. It injects known statistical phenomena into datasets,
validates that each signal is unambiguous, computes ground-truth answers,
evaluates models, and grades their responses.

## Development environment

All commands run from the repository root through `uv`. Dependencies are
declared in `pyproject.toml` and locked in `uv.lock`.

```bash
uv sync
uv run python -m unittest discover -s tests -p "test_*.py"
```

Python 3.11-3.12 is supported; `.python-version` pins 3.11 for local
development. Pipeline scripts use sibling imports, so run them as
`python src/<script>.py` rather than importing the repository as an installed
package.

## Pipeline commands

### Single-config runner

The recommended path is one YAML file:

```bash
cp experiments/example.yaml experiments/my_run.yaml
uv run python src/run_experiment.py experiments/my_run.yaml
uv run python src/run_experiment.py experiments/my_run.yaml --force
```

The runner implements five stages after one-time data preparation:

1. phenomenon injection
2. validation
3. answer computation
4. model evaluation
5. grading

Its stage flags are `skip_phenomena`, `skip_validate`, `skip_answer`,
`skip_eval`, and `skip_grade`. `--force` overrides the config's `force` field
and re-runs validation.

A fresh Stage 1 run deliberately creates manifests without validation or
answers. If evaluation is enabled, the same invocation may not skip Stages 2
or 3. When Stage 1 is skipped, the runner reuses only a matching run-scoped
manifest batch. Grading in the combined runner consumes only evaluation output
created by that invocation; use `grade_pipeline.py` directly to re-grade an
existing result.

### Individual stages

```bash
# Stage 1: inject phenomena
uv run python src/phenomena_pipeline.py --seed 42
uv run python src/phenomena_pipeline.py \
  --summary data/standardized/summaries/bike_sharing_100.json \
  --seed 42
uv run python src/phenomena_pipeline.py \
  --template fc_nonmonotone_peak_v0 fc_interaction_dominant_v0
uv run python src/phenomena_pipeline.py \
  --summary data/standardized/summaries/bike_sharing_100.json \
  --question-type business

# Stage 2: validate
uv run python src/validate_pipeline.py
uv run python src/validate_pipeline.py --dataset bike_sharing_100
uv run python src/validate_pipeline.py --injector fc_nonmonotone_peak
uv run python src/validate_pipeline.py --force

# Stage 3: compute ground truth
uv run python src/answer_pipeline.py

# Stage 4: evaluate; --models is required
uv run python src/eval_pipeline.py \
  --models gpt-4o-mini \
  --table-in-prompt \
  --output data/results/eval_results_gpt-4o-mini.json
uv run python src/eval_pipeline.py \
  --models gpt-4o-mini \
  --dataset bike_sharing_100 \
  --tools load_data run_python

# Stage 5: grade
uv run python src/grade_pipeline.py \
  --input data/results/eval_results_gpt-4o-mini.json
uv run python src/grade_pipeline.py \
  --input data/results/eval_results_gpt-4o-mini.json \
  --model gpt-4o
```

### Manual questions

`src/manual_pipeline.py` creates an already-validated instance from a user CSV
and hand-authored QA pairs, bypassing Stages 1-3. `manual/example.yaml` is the
schema-by-example.

```bash
uv run python src/manual_pipeline.py manual/example.yaml
uv run python src/eval_pipeline.py \
  --models gpt-4o-mini \
  --dataset bike_sharing_100_manual \
  --injector manual \
  --tools run_python \
  --output data/results/eval_results_bike_sharing_100_manual_gpt-4o-mini.json
uv run python src/grade_pipeline.py \
  --input data/results/eval_results_bike_sharing_100_manual_gpt-4o-mini.json
```

Manual question IDs must be unique and must not collide with registered
template IDs. `--force` cleanly recreates the selected manual instance. The
normal answer stage reports manual IDs as `SKIP` and preserves authored
answers.

## Architecture

### Stage 0: data preparation

`scripts/standardize_csvs.py` turns prepared tables under `data/base/` into
row-count variants under `data/standardized/`. The default sizes are 100, 500,
and 1000; the repository also tracks 10,000-row variants.

```bash
uv run python scripts/standardize_csvs.py --sizes 100 500 1000 10000
uv run python scripts/make_summaries.py --rewrite
```

`scripts/make_summaries.py` records the target, inferred dtypes/kinds, and
sampled unique counts used by template matching. Dataset sources, licenses,
and base-table preparation steps are documented in `data/DATASETS.md`.

### Stage 1: template matching and injection

`src/phenomena_pipeline.py` and `src/find_applicable_templates.py` load JSON
templates, match their constraints against dataset summaries, resolve slots,
and call the selected phenomenon injector.

Important invariants:

- one phenomenon per instance directory
- the injector modifies a copy of the source table
- QA answers are initially null
- `manifest.json` is the source of truth for metadata, validation, QA pairs,
  and answers
- the same injector plus resolved parameters is deduplicated
- slot-selection RNGs derive per template from the seed, summary content and
  template ID; injection RNGs derive per deduplicated injector spec from the
  seed, injector name and resolved parameters; neither depends on checkout
  paths or template discovery order
- `InjectionRejected` is an expected dataset rejection; configuration errors,
  broken return contracts, and unexpected exceptions are fatal batch errors
- runner manifests include a content fingerprint covering question type,
  summaries, source tables, template selection, and template contents
- reuse requires an exact fingerprint match
- a successful rebuild removes obsolete instances in the same run/seed scope

Runner output lives under:

```text
data/instances/runs/<run_id>/<dataset>/seed_<N>/<instance_name>/
```

Standalone runs without `--run-id` retain the legacy unscoped layout.

### Stage 2: validation

`src/validate_pipeline.py` performs analytical checks on the modified raw
table: dominance tests, distribution comparisons, interaction checks, and
other phenomenon-specific invariants. It writes a `validation` block to the
manifest.

Revalidation clears the previous result before loading data or invoking the
validator. A missing validator invalidates an old pass. Authored validation on
manual instances is preserved. A normal failed check exits successfully; a
processing error is fatal.

### Stage 3: answer computation

`src/answer_pipeline.py` dispatches each QA pair to the answer computer
registered for its template ID. Ground truth is computed from the modified
table, slot assignments, and the injector's recorded effects.

Only validated instances are eligible. Missing, unimplemented, or failed
answer computers clear stale computed answers. Unexpected errors are collected
across manifests and fail the stage after its summary is printed. Authored
manual answers are preserved.

`fc_nonmonotone_peak_v0` computes its answer from the modified table rather
than from the injected centre: the observed peak is the mean argmax of a
Gaussian-kernel local-linear smoother and a tricube local-quadratic smoother,
rounded to the nearest integer for integer-valued features. Its verifier
raises `AnswerUnavailable` (in `src/phenomena/_base.py`) when the observed
peak cannot be verified; the answer stage then reports `NO_GOLD`, leaves the
answer null without failing, and evaluation skips the QA pair. Stage 2
validation is unaffected.

### Stage 4: model evaluation

`src/eval_pipeline.py` evaluates each eligible `(instance, QA, model)` tuple.
The default context is question-only. Prompt and tool precedence is:

```text
load_data > columns_only > table_in_prompt > question-only
```

Only two model tools are supported:

- `load_data`: returns the current instance CSV
- `run_python`: executes model-authored Python in a per-QA/model subprocess
  with the instance table preloaded as `df`

Evaluation details:

- `--models` is required
- provider detection occurs when Stage 4 starts
- provider clients are initialized lazily
- invalid or duplicate models fail before model work
- QA pairs whose answer is null are skipped, even inside validated manifests;
  the manifest's other QA pairs are still evaluated
- progress is atomically checkpointed to `*.json.partial`
- the final JSON is replaced only after the full batch succeeds
- an empty successful batch publishes `[]`
- `--injection-delay-seconds` sleeps between instance manifests, never between
  QA pairs within one manifest
- setup, DataFrame verification, and user code share one effective deadline

Each tool-enabled session is bounded by model-call, tool-call, token,
tool-result-byte, and wall-time budgets. Provider SDK retries are disabled and
individual provider calls have their own timeout. Budget-triggered results
record the stop reason and consumption metrics.

Runner outputs are written under:

```text
data/results/runs/<run_id>/eval_results_<run_name>.json
```

### Stage 5: grading

`src/grade_pipeline.py` grades entries as `CORRECT`, `PARTIAL`, or `INCORRECT`.
Errors and empty answers are deterministic `INCORRECT` results. Canonical,
unambiguous `value` and leading `column_value` answers use a 5% relative
tolerance. Free-form explanations, negations, contrasts, alternatives, and
additional paired values defer to the OpenAI judge.

The implementation currently requires `OPENAI_API_KEY` before processing,
even when every entry is deterministic. Grading uses marked partial
checkpoints and atomic publication. Runner outputs are placed in the run's
`graded/` directory.

## Experiment config rules

`experiments/example.yaml` documents the full surface. Important constraints:

- `name`, `run_id`, and every `runs[].name` are NFC-normalized portable path
  components of at most 128 characters
- run names must remain unique after normalization and case folding
- a requested `run_id` may not alias an existing instance/result namespace by
  case or normalization spelling
- configure exactly one of `summary` and `summaries_dir`
- top-level `models` are defaults; each effective eval run must resolve to at
  least one model
- `runs[].tools` defaults to `[]` and accepts only `load_data` and
  `run_python`; there is no top-level tools fallback
- Validation treats `dataset_filter` as a substring; Evaluation treats it as a
  suffix
- Validation exact-matches injector filters; Evaluation substring-matches
  them
- Answer computation operates on the selected manifest batch and relies on
  the validation gate

When `skip_phenomena` is enabled, reuse requires the matching `run_id`, seed,
question type, summaries, source tables, templates, and template contents.
Legacy or mismatched manifests must regenerate Stage 1.

## Model-family registry

`MODEL_FAMILIES` in `src/eval_pipeline.py` is the single source of truth for
provider detection, token-limit keyword selection, and extended-thinking
capabilities. Order matters: longer prefixes must precede broad fallbacks.

Current families cover:

- Anthropic Claude Fable, Opus, Sonnet, Haiku, and broad Claude fallbacks
- OpenAI `gpt-*`, `o1`, `o3`, and `o4`
- DeepSeek API `deepseek-*`, including thinking/tool handling for V4
- Amazon Bedrock's configured DeepSeek route and generic `bedrock:` IDs

Use `detect_provider(model)`, `_openai_tokens_kwarg(model, n)`, and
`supports_thinking(model)` rather than duplicating prefix logic. Unknown model
prefixes must raise with the known-prefix list.

DeepSeek V4 tool requests omit unsupported `tool_choice`, preserve
`reasoning_content`, and replay non-null assistant content across rounds.

## Phenomenon registry

Each phenomenon module under `src/phenomena/` exports a frozen `Phenomenon`
binding with:

```python
Phenomenon(
    name="injector_name",
    inject=inject,
    validate=validate,
    compute_answers={"template_id": compute_answer},
)
```

`src/phenomena/__init__.py` exposes:

- `PHENOMENA`: injector name to `Phenomenon`
- `TEMPLATE_TO_PHENOMENON`: template ID to `Phenomenon`

Template-ID collisions fail at import time. Validators belong to injectors,
while answer computers belong to template IDs; one injector may therefore
serve multiple question variants.

### Canonical phenomenon/template map

| Phenomenon | Template IDs | Category |
|---|---|---|
| Bad row indicator | `dq_bad_row_indicator_v0` | Data Quality |
| Conditional-error row indicator | `dq_bad_row_indicator_v1` | Data Quality |
| Unreliable feature | `dq_unreliable_feature_v0` | Data Quality |
| Categorical target outlier | `dq_categorical_target_outlier_v0` | Data Quality |
| Conditional bad rows | `dq_conditional_bad_rows_v0` | Data Quality |
| Underpowered group evidence | `dq_group_evidence_underpowered_v0` | Data Quality |
| Missing-like category label | `dq_missing_label_semantic_v0` | Data Quality |
| Target-dependent missing label | `dq_missing_label_target_dependent_v0` | Data Quality |
| Noise feature | `fc_noise_feature_v0` | Feature Contribution |
| Non-monotone peak | `fc_nonmonotone_peak_v0` | Feature Contribution |
| Monotone classification | `fc_monotone_classify_v0` | Feature Contribution |
| Threshold value | `fc_threshold_value_v0` | Feature Contribution |
| Dominant interaction | `fc_interaction_dominant_v0`, `fc_interaction_direction_v0` | Feature Contribution |
| Pairwise interaction | `fc_pairwise_positive_synergy_v0`, `fc_pairwise_antagonistic_reversal_v0`, `fc_pairwise_compensatory_reversal_v0` | Feature Contribution |

### Interfaces

Injector:

```python
def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    ...
```

The metadata dict contains `type`, `params`, and `effects`.

Validator:

```python
def validate(
    df: pd.DataFrame,
    effects: dict,
    target_col: str,
) -> ValidationResult:
    ...
```

Answer computer:

```python
def compute_answer(
    df: pd.DataFrame,
    slot_assignments: dict,
    effects: dict,
) -> Any:
    ...
```

### Adding a phenomenon

1. Add `src/phenomena/<name>.py` with injector, validator, answer computer(s),
   and a final `PHENOMENON` binding.
2. Import it and add it to `_MODULES` in `src/phenomena/__init__.py`.
3. Add one or more templates under `templates/<category>/`.
4. Run `uv run python scripts/validate_templates.py`.
5. Add focused injector, validator, answer, and reproducibility tests.

Shared analytical helpers belong in `src/shared/metrics.py` or another focused
module under `src/shared/`.

## Template system

Templates under `templates/<category>/` define:

- slots and their sources (`dataset.target`, column values, or defaults)
- dtype/kind/cardinality constraints
- dataset-level requirements
- feature-pool exclusion rules
- injector parameter mappings
- data-science and business question wording
- answer format

The schema is `templates/template_schema.json` (JSON Schema draft 2020-12).
Validate all templates with:

```bash
uv run python scripts/validate_templates.py
```

## Result analysis

```bash
uv run python scripts/eval_stats.py data/results/eval_results.json
uv run python scripts/split_results_by_template.py results.json outdir
```

## Environment

API keys belong in the git-ignored `.env` file:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`

Bedrock uses the standard AWS credential chain. Region selection checks
`BEDROCK_REGION`, `AWS_REGION`, then `AWS_DEFAULT_REGION`. Optional Bedrock
model-ID and token-cap settings are documented in `.env.example`.

Generated instances, raw snapshots, result files, and local environments are
git-ignored. Never commit credentials or generated model output.
