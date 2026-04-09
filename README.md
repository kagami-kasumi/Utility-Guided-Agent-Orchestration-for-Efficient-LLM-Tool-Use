# Agent Policy Pilot (HotpotQA)

可复现对比四类 agent policy：

- `direct`：不调用工具直接回答
- `workflow`：固定调用一次 BM25 search 再回答
- `react`：ReAct（`search/answer` 迭代）
- `threshold`：基于 `uncertainty * expected_gain - cost_weight * step_cost >= threshold` 决策是否搜索

所有方法共享：

- 同一模型（OpenAI-compatible chat 接口）
- 同一检索工具（本地 BM25）
- 同一抽样集（固定 seed）

并输出：

- EM / F1
- tokens
- tool_calls
- wall_time
- 质量-成本 Pareto 图
- 每条样本执行轨迹 JSONL（便于误差分析）

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python scripts/run_pilot.py \
  --hotpot_dev_path /path/to/hotpot_dev_distractor_v1.json \
  --corpus_jsonl_path /path/to/local_corpus.jsonl \
  --output_dir outputs/pilot_run1 \
  --model_name your-model-name \
  --model_base_url http://127.0.0.1:8000/v1 \
  --model_api_key EMPTY \
  --sample_size 200 \
  --seed 42 \
  --topk 3 \
  --max_tool_calls 1,3 \
  --methods direct,workflow,react,threshold
```

`--corpus_jsonl_path` 可选；不传时会默认从 Hotpot dev 的 `context` 构建 BM25 语料。

## 输出目录

- `run_config.json`：完整实验配置
- `sampled_ids.json`：固定抽样 ID 列表
- `summary.json` / `summary.csv`：方法级汇总指标
- `per_sample_metrics.csv`：逐样本指标
- `pareto.png`：F1-成本图（tokens / wall_time）
- `traces/*.jsonl`：逐样本执行轨迹

## 扩展实验（Task 1-5）

```bash
python scripts/run_experiments.py \
  --hotpot_dev_path /path/to/hotpot_dev_distractor_v1.json \
  --output_dir outputs/expanded \
  --primary_model llama-3.1-8b-local \
  --secondary_model qwen2.5-14b \
  --model_base_url http://127.0.0.1:8000/v1 \
  --model_api_key EMPTY \
  --sample_size 200 \
  --react_steps 1,2,3,4,5 \
  --token_budgets 400,600,800,1000,1500,2000
```

主要输出：

- `results.json` / `results.csv`
- `table_react_steps.csv`
- `budget_experiment.csv`
- `utility_ablation.csv`
- `policy_ablation.csv`
- `model_generalization.csv`
- `figures/*.png|*.pdf|*.svg`
- `tables/table_main_results.tex`
- `tables/table_ablation.tex`
- `tables/table_budget.tex`

## 术语说明

- `step_cost`：policy 内部使用的归一化步代价，当前默认定义为 `((current_tool_calls + 1) / max_tool_calls)`。
- `token_usage` 和 `wall_time`：评测输出指标，用于报告成本；默认 `policy` 不直接把它们作为决策输入。
- `expected_gain`：LLM self-estimated marginal value of one more retrieval step。
- `uncertainty`：LLM self-estimated uncertainty of the current answer。
- `expected_gain` 与 `uncertainty` 都会被裁剪到 `[0,1]`，它们是 heuristic signals，不是 calibrated probabilities。

## 增补实验（316）

```bash
python scripts/run_supplement_316.py \
  --hotpot_dev_path /path/to/hotpot_dev_distractor_v1.json \
  --base_results_dir outputs/expanded \
  --output_dir outputs/supplement_316 \
  --primary_model llama-3.1-8b-local \
  --model_base_url http://127.0.0.1:8000/v1 \
  --model_api_key EMPTY
```

该补充脚本会：

- 复用 `outputs/expanded` 的主结果和抽样集
- 额外运行 `policy(token_cost)`、`policy(latency_cost)`、`policy(semantic redundancy)`、`workflow-search-twice`、`workflow-search-verify`
- 导出：
  - `table_main_results.csv`
  - `cost_definition_comparison.csv`
  - `workflow_fairness.csv`
  - `redundancy_comparison.csv`
  - `heuristic_signal_analysis.csv`
  - `tables/*.tex`
  - `figures/figure6_heuristic_signals.*`
  - `实验结果-实验增补表316.md`
