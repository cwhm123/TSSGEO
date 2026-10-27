# TSSGEO

Diagnostic-space document optimization for generative engines, in three parts:

1. **Diagnostic space construction** (`tsgeo/space/`) — a GLM-5 engine annotates
   queries with cited answers; per-document contribution scores and Qwen3-8B
   activations over the same construction set yield a contribution axis and an
   LLM-labeled dimension dictionary.
2. **Bridge to a target engine** (`tsgeo/bridge/`) — a ridge bridge `W` maps
   Llama (target-engine) activations into the frozen Qwen diagnostic space,
   so diagnosis can run on an engine that never built the space.
3. **Rewriting** (`tsgeo/rewrite/`, `tsgeo/case/`) — the complete
   diagnosis-grounded rewrite pipeline on the bridged engine, plus the
   case-reuse variant that replaces per-document online grounding with an
   offline case library and a single online call, and the metric evaluation.


## Repository layout

```
tsgeo/
  config.py                roots + engine API settings (env-driven)
  core/                    axis (contribution axis w), diagnose (+grounding), llm (rewrite call)
  space/                   annotate_citations, convert_scores, extract_activations,
                           build_dimension_dict, build_accumulated_dict
  bridge/                  encoder (robust user-span matching), fit (ridge bridge)
  rewrite/                 glm_native, bridged_llama, bridged_metered, eval_metrics
  case/                    common (protocol), build_library, rewrite_cases, compare_analysis
  retrieval_axis/          vendored activation-extraction toolkit (7 modules)
scripts/1_annotate.sh      Part 1 step 1: GLM citation annotation (train + test)
scripts/2_build_space.sh   Part 1 steps 2-5: samples -> acts -> dictionaries
scripts/3_bridge_llama.sh  Part 2: Llama acts (train/test), Qwen test acts, bridge fit
scripts/4_rewrite_llama.sh Part 3a: bridged rewrite + evaluation
scripts/5_case_reuse.sh    Part 3b: case library, metered reference, case rewrite, report
data/                      frozen dictionaries + 5-query demo subset (see Data below)
```

## Setup

```bash
pip install -r requirements.txt          # torch, transformers, accelerate, numpy, scipy, nltk, anthropic

export GLM_API_KEY=<your zhipu key>      # annotation, dictionary labeling, grounding, rewriting, evaluation
```

Models (any HF format local path; adjust via env):

```bash
export TSGEO_QWEN_MODEL=models/Qwen3-8B           # probing model (diagnostic space)
export TSGEO_LLAMA_MODEL=models/Meta-Llama-3.1-8B # target engine
```

Optional alternative rewriter/evaluator engines: `export GPT_API_KEY=...` or
`export CLAUDE_API_KEY=...` and pass `--rewriter-engine gpt|claude` /
`--engine gpt|claude`. The default and paper pipeline is GLM-5.

## Data

Two roots, both env-overridable:

| env | default | holds |
|---|---|---|
| `TSGEO_DATA_ROOT` | `<repo>/data` | queries, labels, samples, activations, dictionaries, outputs |
| `TSGEO_MODEL_ROOT` | `<repo>/models` | model checkpoints |

Put the full query sets under `data/researchy/` (train) and
`data/researchy/test_queries/` (test)。

Query files (AutoGEO style, one per query):

```json
{"query_id": "ag242656", "query": "...",
 "documents": [{"document_idx": 0, "rank": 1, "url": "...", "title": "...",
                "domain": "...", "text": "..."}]}
```

Activation files are dicts `{sample_id: Tensor(n_layers, hidden)}` in bf16,
where `sample_id = f"{query_id}_doc_{idx}"`.

## Quickstart (demo, end to end in minutes)

The demo runs the full chain on 5 queries / 25 construction documents; only the
activation steps need a GPU (minutes), and only steps that call GLM-5 need the
API key.

```bash
export GLM_API_KEY=...
export TSGEO_DATA_ROOT=data/demo
export TSGEO_N=5

bash scripts/2_build_space.sh            # ~25 docs through Qwen + dictionary build
bash scripts/3_bridge_llama.sh           # ~25 train + ~50 test docs through Llama + fit
bash scripts/4_rewrite_llama.sh          # 5 bridged rewrites + evaluation
bash scripts/5_case_reuse.sh             # case library (5 cases) + comparison
```

Outputs land under `data/demo/researchy/` (`rewrites_bridge_llama.jsonl`,
`*_evalmetrics.json`, `bridge/casecmp/RESULTS.md`).

## Full run

Populate `data/researchy/` with the full query sets (layout under Data
above), then run the five scripts in order. Canonical parameters are the script
defaults: random target selection with seed 42, `--resume` everywhere, GLM-5
for every LLM step. Per-step entry points and flags:

```bash
python -m tsgeo.space.annotate_citations --documents-dir ... --output-jsonl ... --resume
python -m tsgeo.space.convert_scores --labels ... --queries-dir ... --output ...
python -m tsgeo.space.extract_activations --model <ckpt> --output .../all.pt --gpu 0
python -m tsgeo.space.build_dimension_dict
python -m tsgeo.space.build_accumulated_dict
python -m tsgeo.bridge.fit --target llama
python -m tsgeo.rewrite.glm_native --n 100 --target-mode random --seed 42 --resume
python -m tsgeo.rewrite.bridged_llama --probe llama --n 100 --target-mode random --seed 42 --resume
python -m tsgeo.rewrite.eval_metrics --rewrites ... --gen-before --workers 6 --resume
python -m tsgeo.case.build_library --n-cases 1200 --resume
python -m tsgeo.rewrite.bridged_metered --probe llama --n 100 --resume
python -m tsgeo.case.rewrite_cases --probe llama --n 100 --resume
python -m tsgeo.case.compare_analysis --conds cond4_llama llama_test_sim_k3
```
