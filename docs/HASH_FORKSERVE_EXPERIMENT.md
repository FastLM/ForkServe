# HashForkServe / APP — 完整实验

对照设计说明：[`HASH_FORKSERVE.md`](HASH_FORKSERVE.md)。控制面：[`ARCHITECTURE.md`](ARCHITECTURE.md) §10。vLLM 映射：[`FORKSERVE_VLLM_DESIGN.md`](FORKSERVE_VLLM_DESIGN.md)。

新方法是两层：

* **HashForkServe**：会话内 `fork` 仍是 O(1) 别名；`commit` 之后把已提交的整页发布进 APC 哈希索引，下一会话才能命中。投机页在 LCP / `commit` 之前不进哈希。
* **APP**（Advanced Prefill Pruning，系统名 `forkserve_plus`）：在 CoW 之上按便宜到贵过滤 prefill。层序是 hash skip → draft / early prune → 只对 miss tail 做 GPU prefill → disagg 只 `insert` 幸存者。

实验分三档。前两档不需要 GPU，数字来自令牌代价模型，用来核对控制面不变量。第三档才是墙上时间、显存和任务分数。

| 档 | 回答什么 | 需要 |
|---|---|---|
| 0 单测 | 哈希隔离、剪枝、disagg 门控 | CPU |
| 1 控制面 | 五方法 fan-out / replay、峰值 KV、合成并发 | CPU |
| 2 GPU | `vllm_apc` / `forkserve` / `forkserve_plus` 的延迟、峰值 KV、准确率 | 权重 + 数据集 + GPU |

## 0. 环境

仓库根目录（下文命令都从这里跑）：

```bash
cd /path/to/ForkServe
python -m pip install -e ".[dev]"
export PYTHONPATH=.
```

GPU 档再装 vLLM，并准备权重与数据：

```bash
python -m pip install -e ".[vllm]"
export FORKSERVE_ENV="${FORKSERVE_ENV:-$HOME/envs/forkserve}"
export FORKSERVE_MODEL="${FORKSERVE_MODEL:-$HOME/models/Qwen3-8B}"
export FORKSERVE_BENCHMARKS="${FORKSERVE_BENCHMARKS:-$HOME/benchmarks}"
bash scripts/download_benchmarks.sh
```

`download_benchmarks.sh` 写入 `$FORKSERVE_BENCHMARKS/{gsm8k,humaneval,game24}`。数学集（svamp、gsmhard、math500、aime、amc23）若本地没有对应 jsonl，从 `--workloads` 里拿掉即可。

控制面代价（`experiments/prefill_prune_bench.py` 里写死，改数字前先改脚本）：

| 旋钮 | 值 | 含义 |
|---|---|---|
| `page_size` | 16 | 哈希与 CoW 的页 |
| `prefill_us_per_token` | 12 | prefill ms = tokens × 12 / 1000 |
| `disagg_transfer_us_per_token` | 2 | 传输 ms = 发出的 tokens × 2 / 1000 |
| trunk / residual | 256 / 32 | ToT 主干与每条思路 |
| sessions × branching | 10 × 4 | 第一阶段 fan-out |
| replays | 20 | 第二阶段重放已提交赢家 |

APP 开关来自 `plus_config()`（`--system forkserve_plus` 走同一组）：`lazy_abort`、`pointer_swap`、`prune_enabled`、`hash_prune`、`disagg_prefill`，`spec_pool_frac=0.25`，`decode_stop=("####", "\\boxed", "</think>")`。草稿剪枝阈值 `prune_threshold=0.15`，早停看残差前 `early_prune_frac=0.20`。赢家下标 0 始终保留。

## 1. 单测（档 0）

```bash
python -m pytest tests/test_hash_forkserve.py tests/test_prefill_prune.py -q
```

通过条件：全部绿。这两份测试锁住的行为是：

* 投机页在 `commit` 之前不进入 `HashPageIndex`。
* `fork` 不在关键路径上做哈希遍历。
* `commit` 后整页可被下一次 `open` 命中；`cache_salt` 把租户隔开。
* 已发布前缀被 `HASH_SKIP`（重放 0 个 prefill token）或 `HASH_PARTIAL`（只算残差）。
* 非法 / 循环残差被 draft skip；赢家保留。
* `DisaggPrefillConnector.gate` 只把幸存者送进 P/D 管道。

## 2. 控制面套件（档 1）

一条命令跑完哈希微基准、五方法对比和 ForkServe+ mock：

```bash
mkdir -p logs/hash_app
PYTHONPATH=. python experiments/hash_forkserve_bench.py
PYTHONPATH=. python experiments/prefill_prune_bench.py
OUT=logs/hash_app/eval_plus.json bash scripts/run_eval_plus.sh
```

没有 `scripts/run_eval_plus.sh` 里的 venv 时，直接：

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from forkserve.eval_plus import run_suite
r = run_suite(Path('logs/hash_app'))
print('wrote', r.get('wrote'))
"
```

### 2.1 哈希微基准

`experiments/hash_forkserve_bench.py` 写 `experiments/hash_forkserve_bench.json`。三种模式：

| 模式 | 负载 | 要看到的 |
|---|---|---|
| `apc_only` | 40 个独立会话，共享 256 token 主干 | `hash_hits > 0`，`fork_aliases = 0`，活页高于 CoW |
| `cow_fork_only` | 10 个父节点 × 4 路 `fork`，不重放 | `fork_aliases > 0`，跨会话 `hash_hits = 0` |
| `hash_forkserve` | 同上 fan-out，`commit` 后 20 次重放 | 活页接近 CoW，同时 `hash_hits > 0` |

参考快照（`page_size=16`，控制面，不是 GPU 计时）：

| 模式 | live pages | hash hits | fork aliases |
|---|---:|---:|---:|
| APC only | 96 | 624 | 0 |
| CoW fork only | 56 | — | 640 |
| HashForkServe | 56 | 484 | 640 |

Hybrid 同时保住 CoW 的活页数和 APC 的跨会话命中。`wall_ms` 是 Python 控制面耗时，不要拿去和 GPU TTFT 比。

### 2.2 五方法 prefill

`experiments/prefill_prune_bench.py` 写 `experiments/prefill_prune_bench.json`。每个方法两行：`fanout`（第一次 ToT）和 `replay`（重放已发布赢家）。

stdout 列：`prefill_tok`、`prefill_ms`、`xfer_tok`、`fanout_ms`、`peak_kv`、`prune`。

| 方法 | 第一次 fan-out 在做什么 |
|---|---|
| `apc` | 哈希在 token 存在之后才建。同一批 fan-out 尚未发布，主干克隆 Θ(k)。不剪枝。 |
| `forkserve` | `fork` 别名主干；四条残差都 prefill；同步 abort。 |
| `hash_prefill` | 只有 APC 索引（`HashForkServe.materialize`）。第一次 fan-out 仍付克隆；replay 跳过已发布整块。 |
| `disagg_prefill` | prefill 与 APC 相同，然后把全部活 KV 发给 decode。库存 disagg 不剪枝，所以 `xfer_tok = prefill_tok`。 |
| `app` | CoW + hash skip + draft/early prune + 只运送幸存者 + lazy abort。 |

参考快照（10×4，trunk 256，residual 32，只列 `fanout`）：

| 方法 | prefill tok | prefill ms | xfer tok | fanout ms | peak KV | pruned |
|---|---:|---:|---:|---:|---:|---:|
| APC | 11520 | 138.2 | 0 | 139.0 | 11520 | 0 |
| ForkServe | 3840 | 46.1 | 0 | 47.1 | 3840 | 0 |
| hash_prefill | 11520 | 138.2 | 0 | 139.0 | 11520 | 0 |
| disagg_prefill | 11520 | 138.2 | 11520 | 162.1 | 11520 | 0 |
| APP | 2912 | 34.9 | 352 | 35.9 | 2880 | 20 |

核对（脚本打印的相对量，允许浮点末位差）：

* APP vs APC：prefill −74.7%，fan-out −74.2%，peak KV −75.0%。
* APP 的 peak KV = 10 × (256 + 32) = 2880（只留主干加赢家残差）。APC peak = 10 × 4 × (256 + 32) = 11520。
* `pruned = 20`：10 个会话 × 2 条无望思路（循环、非法）。
* replay 阶段 APP 的 `prefill_tokens = 0`（整段已发布提示被 hash skip）。`hash_prefill` 的 replay 命中主干，但第一次 fan-out 没有 CoW。
* `disagg_prefill` 的 fan-out ms 高于 APC，差额是 11520 token 的传输。APP 只运 352 token。

JSON 里另外两块不要当成 GSM8K 实测：

* `decode_curve`：50 题合成标签（42 对 8 错，目标准确率 84.5%），APC `stop_frac=1.0`（每题 256 token）对 APP `stop_frac=0.6`。参考：达到 84.5% 时 APC 256 token、APP 153 token。这是早停模型，不是阅卷。
* `concurrency`：用峰值 KV 和式 (9)（Θ ∝ C / T_e2e，C ∝ HBM / M）做 QPS 扫描。默认 KV 池 8 GiB、每 token 147456 字节、e2e 0.4 s、网格 8…256 QPS。参考：P99 TTFT ≤ 1 s 的最大 QPS 从 192 到 256，tokens/s +35.1%。`sweep[]` 里看 `apc_p99_ttft_s`、`fs_p99_ttft_s`、`apc_slo_ok`、`fs_slo_ok`。

### 2.3 Mock fan-out（`forkserve.eval_plus`）

`logs/hash_app/eval_plus.json`（或 `OUT` 指向的文件）对比 `micro_baseline`（ForkServe）和 `micro_plus`（APP）。要看的字段：

| 字段 | APP 相对基线 |
|---|---|
| `pruned_branches` | > 0 |
| `prefilled_branches` | < branching |
| `hash_skips` | replay / 已发布主干上 > 0 |
| `early_aborts` | 残差前缀已经无望时 > 0 |
| `transfer_tokens` | 小于「全部活 KV」 |
| `abort_mark_ms` | lazy abort 标在 TTFT 之外；`abort_reclaim` 不进 fan-out |
| `pointer_swaps` | fan-out 是 incref，不是 KV memcpy |
| `peak_kv_tokens` | 低于基线（输家页不进峰值） |

`max_qps_p99_1s` 与 `tok_s_gain` 与 2.2 的并发扫描同一模型。脚本 stdout 会打 peak tok/s 和 P99≤1s 的 QPS。

## 3. GPU 端到端（档 2）

`--system forkserve_plus` 打开 APP。对照是 `vllm_apc`（跨请求哈希）和 `forkserve`（CoW，不剪枝）。`vllm_recompute` 只在需要「相对重算的加速比」时加上；`format_table` 的 `vs_recompute` 列靠它。

编排器按可见 GPU 数选 TP：≥4 张卡默认 TP=2 和 TP=4，≥2 张默认 TP=2，否则 TP=1。每个 `(system, tp)` 起一个 worker，写 `logs/hash_app/bench_<system>_tp<tp>.json`，最后合并到 `--out`。worker 会等目标卡占用降到约 4 GiB 以下；卡被占满时该 shard 跳过。

进度：`--progress-log logs/hash_app/progress.log`。

### 3.1 冒烟（mock 后端，确认参数）

不加载权重。用来确认 workload 列表和 JSON 模式，不代表 GPU 数字。

```bash
PYTHONPATH=. python -m forkserve.bench \
  --backend mock \
  --systems vllm_recompute,vllm_apc,forkserve,forkserve_plus \
  --workloads gsm8k,game24,humaneval,tot,react \
  --limit 4 \
  --decode 64 \
  --branching 4 \
  --out logs/hash_app/mock.json
```

表头：

```
system  tp  workload  e2e_ms  fanout_ms  ttft_obs_ms  peak_kv  task_score  metric  kv_save  vs_recompute
```

`forkserve_plus` 行上 `pruned_branches`、`hash_skips`、`transfer_tokens` 在 JSON 里（表上不打印）。mock 的 `task_score` 不是模型准确率。

### 3.2 主实验

四张卡、Qwen3-8B、TP=2 的默认规模与 `scripts/run_math_apc_vs_forkserve.sh` 对齐，并加上 `forkserve_plus`。GSM8K / SVAMP / GSM-Hard 每题解码至少 512（`--decode` < 64 时保持小值，给冒烟用）。MATH-500 / AIME / AMC23 至少 768。

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PYTHONPATH=. python -u -m forkserve.bench \
  --model "$FORKSERVE_MODEL" \
  --systems vllm_apc,forkserve,forkserve_plus \
  --tp "${FORKSERVE_TP:-2}" \
  --workloads gsm8k,game24,humaneval \
  --limit "${LIMIT:-80}" \
  --chunk "${CHUNK:-8}" \
  --decode "${DECODE:-512}" \
  --gsm8k-decode "${GSM8K_DECODE:-512}" \
  --branching "${BRANCHING:-4}" \
  --sessions 2 \
  --trunk-tokens 512 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-batched-tokens "${MAX_BATCHED:-2048}" \
  --gpu-util "${FORKSERVE_GPU_UTIL:-0.90}" \
  --progress-log logs/hash_app/progress.log \
  --out logs/hash_app/eval.json
```

要和 APC 论文表同一组数学题时，把 `--workloads` 换成：

```text
gsm8k,svamp,gsmhard,math500,aime,amc23,game24
```

`--limit 0` 跑完整 jsonl/csv，此时 `--chunk` 必须 ≥ 1（脚本默认 4，上面用 8）。质量对照、不做 fan-out 时加 `--quality-only`：`peak_kv` 和 `fanout_ms` 保持 0，只比单路径生成。

DeepSeek-R1 distill 权重会自动设 `FORKSERVE_CHAT_STYLE=deepseek_r1`（路径里含 `deepseek` 或 `r1-distill`）。其它聊天模板用环境变量 `FORKSERVE_CHAT_STYLE`。

### 3.3 怎么读 GPU JSON

每个 workload 一行，系统间按 `(tp, workload)` 对齐。APP 成立时，同一 workload 上：

1. **峰值 KV**：`forkserve_plus.peak_kv_tokens` ≤ `forkserve` ≤ `vllm_apc` 的第一次 fan-out。`kv_saving` 是相对克隆的节省，不是相对 APC 的哈希命中。
2. **fan-out 拆分**：`fanout_ms ≈ cow_ms + prefill_ms + abort_mark_ms`（外加 disagg 传输）。`abort_reclaim_ms` 在解码之后，不进 TTFT。`pointer_swaps > 0` 且 fan-out 不是按 k 份主干 memcpy。
3. **剪枝没有改赢家语义**：`task_score`（GSM8K exact match、HumanEval pass 等）相对 `forkserve` 不因多剪赢家而崩。`quality_collapsed` 应为 false。`quality_delta` 是相对参照系统的分差。
4. **传输**：`forkserve_plus.transfer_tokens` 只计幸存者。库存 disagg 路径会把 fan-out 的全部活 KV 发出；APP 行应明显更小。`hash_skips` 在重复主干（多 session、chunk 间共享前缀）上增加。
5. **解码令牌**：`decode_tokens` 是各题之和，`decode_per_item` 是每题预算。APP 的 `decode_stop` 可以在 `####` / `\boxed` 处提前停；准确率持平则每题 token 应 ≤ APC。
6. **准确率曲线**（有 `task_correct` 的行）可以事后画，不必重跑：

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from forkserve.eval_plus import curve_from_eval_json, tokens_to_hit_accuracy
p = Path('logs/hash_app/eval.json')
for system in ('vllm_apc', 'forkserve', 'forkserve_plus'):
    c = curve_from_eval_json(p, system, tp=2, workload='gsm8k')
    hit = tokens_to_hit_accuracy(c, 0.5) if c else None
    acc = c[-1]['accuracy'] if c else None
    print(system, 'n', len(c), 'final_acc', acc, 'tokens_to_0.5', hit)
"
```

`tp` 改成实际写入 JSON 的值。曲线点是 `{items, decode_tokens, accuracy, solved}`。

## 4. 产物

| 文件 | 内容 |
|---|---|
| `experiments/hash_forkserve_bench.json` | APC / CoW / hybrid 活页、命中、别名 |
| `experiments/prefill_prune_bench.json` | 五方法 × {fanout, replay}、summary、合成曲线、QPS 扫描 |
| `logs/hash_app/eval_plus.json` | mock 的 baseline vs plus、并发上界 |
| `logs/hash_app/bench_<system>_tp<tp>.json` | 单个 GPU worker |
| `logs/hash_app/eval.json` | 合并表；`table` 字段是 stdout 那张表 |
| `logs/hash_app/progress.log` | 分 chunk 时间戳 |

控制面 JSON 被脚本覆盖在 `experiments/` 下。要留档就把档 1 的三个命令的输出目录改到 `logs/hash_app/`（`prefill_prune` 可在 Python 里 `run_suite(Path('logs/hash_app'))`）。

## 5. 和设计条款的对应

| 设计条款 | 哪一档核对 |
|---|---|
| 投机页不哈希，直到 commit | 档 0 `test_hash_forkserve` |
| `fork` 不走哈希 | 档 0；档 1 `fork_aliases` 在 fan-out，`hash_hits` 在 replay |
| 提交后的整页可被下一会话命中 | 档 1 replay 行 `prefill_tokens` |
| 不满的块不进 APC；残差页 CoW 可以留 | 档 0 `HASH_PARTIAL`；档 1 ForkServe 活页 < APC |
| 无望兄弟在成为哈希键之前丢掉 | 档 1 `pruned`；档 2 `pruned_branches` |
| 库存 disagg 不提高吞吐；APP 少算少传 | 档 1 `disagg_prefill.xfer_tok` vs `app.xfer_tok` |
| 赢家 decode 只看见提交脊 | 档 2 `task_score` 不塌；峰值 KV 不含输家残差 |
| 式 (9) 并发随 M 下降而上升 | 档 1 `concurrency`；档 2 用实测 `peak_kv_tokens` 代入，不要用合成 192→256 代替 |

档 1 的 192→256 QPS 来自默认的 GSM8K 峰值令牌假设（APC 1563、ForkServe 1163）和 8 GiB KV 池，不是这次 GPU 跑出来的。GPU 跑完后用 `eval.json` 里的 `peak_kv_tokens` 重算才是该模型、该 TP 的容量。
