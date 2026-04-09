# Reproduce the Experiments

This document is for reproducing the main experiments and the final-defense supplement experiments in this repository.

## 1. What This Repo Covers

This repository contains:

- the main HotpotQA orchestration experiments
- the expanded experiment runner
- the final-defense supplement experiments:
  - `policy(token_cost)`
  - `policy(latency_cost)`
  - `policy(semantic redundancy)`
  - `workflow-search-twice`
  - `workflow-search-verify`

The generated outputs are written under `outputs/` and are intentionally ignored by git.

## 2. Prerequisites

- Python `3.10+`
- `conda` recommended
- a local OpenAI-compatible inference endpoint
- enough GPU memory to serve the chosen model with vLLM

## 3. Environment Setup

Create and activate a clean conda environment:

```bash
conda create -y -n blind_repo_env python=3.11
conda activate blind_repo_env
pip install -r requirements.txt
```

## 4. Dataset Preparation

Download HotpotQA dev and place it somewhere local:

```bash
mkdir -p data
wget -O data/hotpot_dev_distractor_v1.json \
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

If you already have the file elsewhere, just set:

```bash
export HOTPOT_DEV=/path/to/hotpot_dev_distractor_v1.json
```

Otherwise:

```bash
export HOTPOT_DEV=$PWD/data/hotpot_dev_distractor_v1.json
```

## 5. Model Serving

Set your model path and served model name:

```bash
export MODEL_PATH=/path/to/your/instruct-model
export MODEL_NAME=local-instruct-model
```

### Start command

```bash
mkdir -p outputs
nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --generation-config vllm \
  > outputs/vllm.log 2>&1 &
```

Adjust `--tensor-parallel-size` to match your GPU setup.

### Health check command

```bash
curl -s http://127.0.0.1:8000/health && echo
curl -s http://127.0.0.1:8000/v1/models && echo
```

### Stop command

```bash
pkill -f "vllm serve $MODEL_PATH"
```

## 6. Run the Main Expanded Experiments

```bash
MPLCONFIGDIR=/tmp OPENAI_API_KEY=EMPTY python scripts/run_experiments.py \
  --hotpot_dev_path "$HOTPOT_DEV" \
  --output_dir outputs/expanded \
  --primary_model "$MODEL_NAME" \
  --model_base_url http://127.0.0.1:8000/v1 \
  --model_api_key EMPTY \
  --sample_size 200 \
  --seed 42 \
  --topk 3 \
  --react_steps 1,2,3,4,5 \
  --token_budgets 400,600,800,1000,1500,2000 \
  --main_tool_calls 1,3,5
```

Expected main outputs:

- `outputs/expanded/results.csv`
- `outputs/expanded/results.json`
- `outputs/expanded/traces/*.jsonl`
- `outputs/expanded/figures/*.pdf`
- `outputs/expanded/tables/*.tex`

## 7. Run the Final-Defense Supplement

```bash
MPLCONFIGDIR=/tmp OPENAI_API_KEY=EMPTY python scripts/run_supplement_316.py \
  --hotpot_dev_path "$HOTPOT_DEV" \
  --base_results_dir outputs/expanded \
  --output_dir outputs/supplement_316 \
  --primary_model "$MODEL_NAME" \
  --model_base_url http://127.0.0.1:8000/v1 \
  --model_api_key EMPTY
```

Expected supplement outputs:

- `outputs/supplement_316/supplement_results.csv`
- `outputs/supplement_316/table_main_results.csv`
- `outputs/supplement_316/cost_definition_comparison.csv`
- `outputs/supplement_316/workflow_fairness.csv`
- `outputs/supplement_316/redundancy_comparison.csv`
- `outputs/supplement_316/tables/*.tex`
- `outputs/supplement_316/figures/figure6_heuristic_signals.pdf`

## 8. Optional: Redraw the Pareto Figures

This step is CPU-only and does not require model serving or GPUs.

```bash
MPLCONFIGDIR=/tmp python scripts/redraw_pareto_figures.py
```

Expected outputs:

- `outputs/expanded/figures/figure1_pareto_tokens.pdf`
- `outputs/expanded/figures/figure2_pareto_latency.pdf`

## 9. Paper Assets

The experiment section currently references:

- `outputs/expanded/figures/figure1_pareto_tokens.pdf`
- `outputs/expanded/figures/figure2_pareto_latency.pdf`
- `outputs/expanded/figures/figure3_step_analysis.pdf`
- `outputs/expanded/figures/figure5_policy_ablation.pdf`
- `outputs/supplement_316/figures/figure6_heuristic_signals.pdf`

and these generated table files:

- `outputs/supplement_316/tables/table_main_results.tex`
- `outputs/supplement_316/tables/table_cost_definition.tex`
- `outputs/supplement_316/tables/table_workflow_fairness.tex`
- `outputs/supplement_316/tables/table_ablation.tex`

## 10. How to Tell Whether Reproduction Finished Successfully

Main experiments are finished when:

- `outputs/expanded/results.csv` exists
- `outputs/expanded/figures/figure1_pareto_tokens.pdf` exists
- `outputs/expanded/figures/figure2_pareto_latency.pdf` exists
- `outputs/expanded/tables/table_main_results.tex` exists

Supplement experiments are finished when:

- `outputs/supplement_316/supplement_results.csv` exists
- `outputs/supplement_316/table_main_results.csv` has `7` rows
- `outputs/supplement_316/workflow_fairness.csv` has `3` rows
- `outputs/supplement_316/cost_definition_comparison.csv` has `6` rows
- `outputs/supplement_316/redundancy_comparison.csv` has `2` rows

## 11. Failure Signals

Check these first if something goes wrong:

- `outputs/vllm.log` contains `Traceback`
- `outputs/vllm.log` contains `CUDA out of memory`
- `curl http://127.0.0.1:8000/health` fails
- the run exits before writing `results.csv`

## 12. Safe Progress Checks

```bash
tail -n 50 outputs/vllm.log
ls outputs/expanded
ls outputs/supplement_316
```
