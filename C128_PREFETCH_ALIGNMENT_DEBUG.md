# NPU DSV4：验证 C128 L3 预取提交是否可能落在非完整块边界

## 目的与当前结论

本分支只增加可关闭的诊断日志，不修改查询命中、keys 选择、槽位分配、
节点插入、返回值或释放逻辑，也不会强行制造一个不存在的 L3 命中。

需要在部署中验证的假设：

1. C4、SWA 等池允许一个不在 C128 完整块边界上的 L3 恢复终点。
2. C128 只读取范围内的完整物理页，所有计划对象都读取成功。
3. `_check_hybrid_prefetch_result()` 接受这个结果。
4. `insert_host()` 的目标节点不在完整块边界，`_commit_prefetch()` 仍挂载 C128 数据。
5. 后续 C128 匹配校验接受这个节点，可能让下一次查询从非对齐位置开始。

目前只有代码检查和隔离方法检查支持上述风险，尚未在 NPU + Mooncake 上复现整条链路。
普通缓存插入有 C128 块边界约束，SWA 的有效快照也可能把实际命中限制在对齐位置；
因此发送非对齐长度的请求不等于一定触发问题。

本分支基于 `f353e64e9c`，没有包含原工作区未提交的分配修正、查询单位修正和新单测。
特别注意：本基线的 Mooncake C128 查询 verdict 仍是完整块数量，
所以日志字段名为 `raw_c128_query_verdict`，不要把它直接当作普通页数。

## 部署与日志开关

使用已有部署脚本，在所有推理 worker 启动之前设置：

```bash
export SGLANG_DEBUG_C128_PREFETCH=1
```

继续使用你现有的 NPU DSV4、Mooncake L3 和 MTP 配置；确保环境变量传入容器和所有 rank。
日志级别使用 `info` 或更详细，保留每个进程 / rank 的原始日志，不要只保留 rank 0。
本次先验证缓存链路与节点边界，不做输出精度或吞吐对比。

目标是 host-cache 模式的 `insert_host → _commit_prefetch` 路径：

```text
--hicache-host-memory-mode cache
```

不要为这个问题切到 `buffer_only`；它不走相同的 host 树插入流程，
部分硬件 / 组件配置也不支持该模式。

排查结束后关闭日志并重启 worker：

```bash
unset SGLANG_DEBUG_C128_PREFETCH
```

开启后会增加节点深度遍历和日志开销，不适合拿本次吞吐结果作为正式性能结论。
日志不打印 token IDs、实际 hash 字符串或 tensor 内容。

## 页数单位

以下例子假定启动日志 / 诊断日志确认了：

| 参数 | 数值 | 单位 |
|---|---:|---|
| 普通 `page_size` | 128 | 原始 token / 普通页 |
| C128 `block_pages` | 16 | 普通页 / 完整 C128 块 |
| C128 `slots_per_page` | 16 | 摘要槽位 / C128 物理页 |
| `group_tokens` | 2048 | 原始 token / 完整 C128 块 |

一个 C128 物理页保存 16 条摘要。恢复 35 个普通页对应 4480 tokens，
H16、H32 两个 C128 对象只有 32 条摘要，不能据此声称整个 35 页终点满足 C128 恢复约束。

`*_mod_group` 的单位是 token，表示某位置除以 `group_tokens` 的余数。
第 35 个普通页终点对应余数 `4480 % 2048 = 384`。

## 用现有压测 / 请求工具验证

### 1. 先跑一个对齐的 L3 命中对照

使用一组固定 token IDs / 固定共享前缀，把缓存写入 Mooncake。
保持模型、量化、TP/PP、缓存布局、namespace / salt 等配置一致。
等待实际 L3 backup 完成，不能仅凭请求返回就认定后台写入已经完成。

让下一次请求确实走 L3，而不是全部命中 L1/L2：可用现有部署流程重启推理 worker，
保留 Mooncake 服务及其数据，并使用相同的存储身份配置。
不要清空 Mooncake；也不要假设某个 flush 接口只清 L1/L2，除非部署脚本已确认其行为。

先确认一次 16 / 32 / 48 个普通页边界的恢复：

- `phase=query` 中 `kv_hit_pages > 0`。
- `phase=result` 中 `all_succeeded=True`。
- `phase=publish` 中 `unaligned=False`、`end_mod_group=0`。

这一步用于确认模型、L3 数据、日志开关和提交路径确实生效。

### 2. 尝试非对齐候选范围

从相同的长共享前缀构造不同长度的请求，建议覆盖：

| 期望候选普通页数（从根开始） | 对应逻辑 token 数 | 用途 |
|---:|---:|---|
| 32 | 4096 | 对齐对照 |
| 33 | 4224 | 刚跨完整块 |
| 35 | 4480 | 重点观察例子 |
| 47 | 6016 | 下一个块之前 |
| 48 | 6144 | 下一个对齐对照 |

这些是缓存逻辑范围，不是保证可以直接使用的 HTTP 输入长度。
普通前缀匹配会保留边界 token，MTP / bigram、SWA 尾部限制、已有前缀命中还可能改变范围。
如使用 `/generate` 的 `input_ids`，按你已有工具发送；如使用文本，以实际 tokenizer 和日志为准。
不要以字符数代替 token 数，也不要假设输入正好 4480 tokens 就查询 35 页。

以 `phase=plan` 的 `anchor_tokens`、`prefetch_tokens` 和 `phase=query` 的
`storage_start`、`query_pages` 为准，确认本次真正查询的起点和长度。
最好先单请求、低并发跑对照，避免同一 request ID 重复使用。

还应覆盖有 L1/L2 前缀的情况，例如先保留前 16 页，再查询后面的范围。
这时日志中的 `kv_hit_pages` 是本次后缀页数；绝对结束位置需要加 `storage_start`。
例如起点 2048 tokens、后缀命中 19 页，绝对终点同样是 4480 tokens。

### 3. 如第一阶段出现非对齐挂载，再发共享前缀的后续请求

不要清理刚恢复的树。发送共享该前缀、并继续向后延伸的请求，观察：

- 同一目标节点是否出现 `phase=match_validator accepted=True` 且余数非零。
- 新预取 `phase=plan` / `phase=query` 是否出现 `start_mod_group != 0`。

这能区分“挂载到了非对齐节点”和“这个节点进一步成为了非对齐查询起点”。

## 日志含义与判定

所有诊断行以 `C128_PREFETCH_DIAG` 开头。

| phase | 含义 | 重点字段 |
|---|---|---|
| `plan` | C128 组件创建预取计划 | anchor 深度、候选终点余数、计划块数 |
| `query` | Mooncake 查询返回，TP 同步之前 | request ID、整体命中普通页数、C128 原始 verdict |
| `result` | 实际 IO 完成，跨 rank 结果同步后的完整性检查 | `all_succeeded`、实际 C128 对象数、各池计划 / 完成数 |
| `insert_host` | 普通 host 范围插入结果 | request ID、`inserted_host_node`、是否丢弃 |
| `commit_decision` | C128 决定挂载或释放之前 | 目标绝对深度、槽位数、读取数量、是否已有副本 |
| `publish` | C128 已实际写入目标节点的 `host_value` | `target_id`、`end_mod_group`、`unaligned` |
| `match_validator` | C128 校验接受了一个非对齐节点 | 节点 ID、深度、device/host 数据是否存在 |

`query`、`result`、`insert_host` 用相同 request ID 对应；
`inserted_host_node` 再与 `target_id` / `node_id` 对应。
节点 ID 是进程 / rank 内的标识，需在同一个 worker 日志中关联，不能只凭跨进程相同数字关联。

`result.pools` 的每一项是 `(池名, 计划 key 数, 实际成功对象数)`。
这些是各侧池自己的对象数，不能直接互相取 min 当作普通页数。
`result.sidecar_hit_pages=0` 且 `has_saved_sidecar_range=False` 是未发生 FULL 头部裁剪的正常情况，
不表示 L3 命中为零。
`assume_stored=True` 的重试可能跳过真实查询，因此没有对应的 `phase=query` 行。

### 真正触发的证据

下面是字段示意，不是已经获得的部署日志：

```text
phase=query ... storage_start=0 kv_hit_pages=35 end_mod_group=384 raw_c128_query_verdict=2
phase=result ... completed_tokens=4480 planned_c128_keys=2 loaded_c128_objects=2 all_succeeded=True
phase=insert_host ... inserted_host_node=123 completed_tokens=4480 dropped=False
phase=publish target_id=123 target_tokens=4480 group_tokens=2048 end_mod_group=384 published_objects=2 unaligned=True
```

**`phase=publish ... unaligned=True` 才是非对齐挂载实际发生的直接证据。**
之后若出现同一节点的 `phase=match_validator ... accepted=True`，说明 C128 校验也接受了它。
这仍不是输出精度错误的证明；本轮只验证缓存边界行为。

### 不算触发的情况

- 只有 `plan.candidate_end_mod_group != 0`：仅表示候选范围不对齐，真实命中可能缩短。
- 查询候选 35 页，最终 `kv_hit_pages=32`：可能被 SWA / 状态池限制在完整边界，属于有效负向观察。
- `phase=result all_succeeded=False`：整批恢复被拒绝，不会走本次 host 插入和 C128 挂载。
- `phase=commit_decision` 余数非零，但目标已有副本或读回不足：不一定实际发生新挂载，必须看 `publish`。
- 全部命中 L1/L2，或者没有 C128 transfer：没有走到待验证的 L3 路径。
- 只出现对齐的 `publish`：本轮未复现，不能写成已证明存在 bug，也不能据此证明所有配置都安全。

如果所有非对齐候选都被查询阶段缩回完整块，请保留这些日志和实际 sidecar 配置。
这可能说明上游存储 / SWA 边界已提供约束，下一步应检查这项约束是否覆盖所有支持的模式，
而不是手工修改 Mooncake 对象或强行篡改命中结果来宣称部署复现。

## 收集结果

使用现有部署日志即可，无需额外测试脚本：

```bash
grep 'C128_PREFETCH_DIAG' server.log
grep 'C128_PREFETCH_DIAG phase=publish' server.log
grep 'C128_PREFETCH_DIAG phase=match_validator' server.log
```

请保留分支 commit、启动参数、硬件型号、TP/PP、MTP 配置、page size、
host memory mode、写策略、请求实际 token 长度，以及完整的相关 request ID 日志。

## 本地验证范围

本地是 Windows 环境，缺少 PyTorch / NPU / Mooncake 运行环境。
提交前已通过语法、格式、导入和差异空白检查。
另用实际方法的隔离检查验证了对齐挂载、非对齐挂载、部分读取和重复副本：
开关前后挂载 / 释放结果相同，关闭时不打印日志；开启时能打印预期诊断标记。
查询和结果阶段的日志字段与格式也做了隔离检查。
这些检查没有执行真实 tensor IO、TP collective 或完整树流程，不能代替本说明中的部署验证。
