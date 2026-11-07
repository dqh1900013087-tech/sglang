# NPU C128 预取分配：日志与部署验证

本文配合现有部署和请求脚本使用，不提供额外测试脚本。

## 修改内容

位置：`python/sglang/srt/mem_cache/unified_radix_cache.py` 的
`UnifiedRadixCache._alloc_prefetch_aux_staging()`。

此函数为 Mooncake → CPU 的预取分配接收空间，此时尚未读取摘要。
`BLOCK_GROUPS` 的 key 数量是组件页数，`hit_pages` 是普通逻辑页数，需要先换算单位。

默认 NPU 配置中，一个普通页覆盖 128 个原始 token；一个 C128 页包含 16 个摘要槽位，
覆盖 2048 个原始 token，也就是 16 个普通页。因此 `block_pages=16`。

旧公式：

```python
min(planned_pool_pages, full_hit_pages) * slots_per_page
```

修正后的公式：

```python
min(planned_pool_pages, full_hit_pages // block_pages) * slots_per_page
```

例如计划读取 4 个 C128 页，实际只命中 32 个普通页：

- 旧公式：`min(4, 32) * 16 = 64` 个摘要槽位。
- 正确需求：`min(4, 32 // 16) * 16 = 32` 个摘要槽位。

后续同步 key 虽然会释放多余空间，但不能挽回此前的申请失败或不必要的淘汰。
本次修正数量换算并增加 INFO 日志，没有修改 MemCache。

## 通过日志确认触发条件

仅在旧、新公式的申请量不同时打印，无需开启存储指标：

```text
HiCache BLOCK_GROUPS staging corrected: request_id=... pool=deepseek_v4_c128 hit_tokens=4096 full_hit_pages=32 planned_pool_pages=4 block_pages=16 slots_per_page=16 legacy_slots=64 requested_slots=32 avoided_slots=32
```

| 字段 | 含义 |
|---|---|
| `request_id` | 关联请求；TP 各 rank 可能分别打印 |
| `hit_tokens` | 传给此分配阶段的原始 token 范围 |
| `full_hit_pages` | 上述范围换算成普通页的数量 |
| `planned_pool_pages` | 当前 key/占位符数量，即计划组件页数 |
| `block_pages` | 一个组件页覆盖的普通页数 |
| `slots_per_page` | 组件池每页槽位数 |
| `legacy_slots` | 旧公式在相同输入下会申请的槽位数 |
| `requested_slots` | 修正后的申请量；为 0 时跳过申请 |
| `avoided_slots` | 本次避免多申请的槽位数 |

**判据：`pool=deepseek_v4_c128` 且 `legacy_slots > requested_slots`。**
这说明真实请求遇到了旧代码会多申请的条件，当前代码已改用正确数量。
日志在实际分配前输出，不代表后续分配、读取或恢复一定成功。
它也不代表旧版本一定申请失败：空间充足时，旧版本可能只是暂时多占用空间。

修正后仍无法分配时，还会输出：

```text
HiCache auxiliary staging allocation failed: request_id=... pool=deepseek_v4_c128 requested_slots=32
```

这表示正确数量的申请也失败，需要检查容量和可淘汰空间等因素，不能单凭它判定换算仍有问题。

## 用现有部署脚本验证

### 部署条件

沿用已可运行的 NPU V4 模型、TP、I/O 布局和 Mooncake 连接配置，加载本次代码。
确认以下启动选项；这不是完整启动命令：

```text
--enable-hierarchical-cache
--hicache-storage-backend mooncake
--hicache-host-memory-mode cache
--hicache-write-policy write_through
--hicache-storage-prefetch-policy wait_complete
--log-level info
```

首次使用单 PP stage，避免混入另一个 PP 任务分页信息传递问题；本次未修改该问题。
使用 `cache` 模式，当前 `buffer_only` 初始化不接受 C128 组件。
模型上下文需容纳下述长请求及输出。

### 构造“计划长、L3 命中短”的请求

1. 请求 A：共同前缀 4096 token，加 64 个尾部 token，只生成少量 token。
2. 等待 A 的前缀和组件数据完成 L3 备份。请求响应结束不等于异步备份完成，
   通过现有存储指标或调试工具确认。
3. 在独立测试实例、无在途请求时调用 `/flush_cache`，确认 HTTP 200，清除本地 L1/L2。
   当前实现不会主动删除 Mooncake 对象，但会清理待处理队列，所以必须先完成第 2 步。
   保留 Mooncake 的数据。
4. 请求 B：相同的 4096-token 前缀，加与 A 不同的约 4096-token 后缀。
   此时本地缓存为空，需要查询更长的 L3 范围，但 L3 只有共同前缀。
5. 查看 B 的日志，比较 `legacy_slots` 和 `requested_slots`。

A/B 使用相同 `cache_salt`，每轮换一个新值，避免历史请求让 B 全量命中。
共同前缀必须是相同 token IDs，不能按文本字符数估算。
不要提前用完整 B 预热，否则后半段也可能进入 Mooncake。

理想示例为计划 4 个 C128 页、实际需要 2 页，即 `64 → 32` 个槽位。
请求末尾处理、预取范围限制和组件可恢复位置可能改变实际计数，例如计划只有 3 页。
不要求日志恰好等于示例，关键是 `legacy_slots > requested_slots`。
无需刻意把整个 CPU 池压缩到 32 个槽位才能观察这个条件。

日志检索：

```bash
rg 'BLOCK_GROUPS staging corrected|auxiliary staging allocation failed' server.log
```

## 没有日志时检查

1. 服务是否加载修改后的源码，是否允许 INFO 日志。
2. A 是否完成 L3 备份，B 是否真正访问 L3，而不是命中本地 L1/L2。
3. A/B 是否共享同一个新 salt，B 后半段是否已有 L3 缓存。
4. 是否进入 NPU C128 的 `BLOCK_GROUPS` 独立分配路径，`block_pages` 是否为 16。
5. `host_indices` 已分配或 `indices_from_pool` 非空时，此函数会跳过任务。
6. keys 已缩减到实际数量，或者本次全量命中时，旧、新公式可能相等，不打印修正日志。
7. 共享 host 分配路径本来就按完整块计算，可能没有修正日志。

出现修正日志，证明实际请求满足旧公式多申请的条件；没有日志不能证明问题不存在。
没有分配失败日志，也不能证明整个 L3 恢复成功。
还需通过现有部署测试检查请求完成、实际 L3 读取及输出正确性。
本地已验证分配逻辑，尚未进行 NPU + Mooncake 实机端到端验证。
