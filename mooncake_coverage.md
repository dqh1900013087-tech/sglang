# Ascend 模型的 Mooncake L3 HiCache 覆盖分析

- 审查日期：2026-11-16。
- 代码基准：`56fa7ca6e7e4004aeb935b6bb38d0acf81340510`（2026-11-13）。
- 模型范围：[Ascend Model Tutorials](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_8b) 导航中的 20 个模型，与本仓库 `docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/` 对照。
- 审查方式：逐模型核对架构入口、设备缓存池、NPU/CPU 搬运、host buffer 导出，以及 Mooncake 注册与存取路径。
- 验证边界：静态代码审查；未运行 NPU 推理、Mooncake 服务或端到端测试。下文引用的测试是仓库中已有的测试，不表示本次执行通过。

## 结论与判定口径

20 个模型中，18 个具备有条件的代码接入路径；**MiMo-V2-Flash 和 GLM-5.3-Flash 在当前默认 Ascend HiCache 路径中存在具体阻断**。

“代码路径支持”表示已找到必要缓存状态的设备池、host 池和 L3 存取实现；不等于该型号及所有部署组合均经过实机验证。结论以 DCP=1、常规 KV 格式、兼容的 host layout 为前提，不自动覆盖 MTP、特殊 KV 量化、统一内存、HiSparse、PD decode offload 或其他可选组合。

审查链路为：

```text
模型 attention / state
  -> device KV / state pool
  -> NPU <-> CPU HiCache 搬运
  -> Mooncake buffer 注册、写入、读取
  -> 完整恢复 KV / indexer / recurrent state
```

本报告讨论 `--enable-hierarchical-cache` 配合 `--hicache-storage-backend mooncake`。
`--disaggregation-transfer-backend mooncake` 的 PD 传输，以及 unified-cache external linker，是不同路径，不能作为本报告的支持证据。

## 逐模型分类

| 序号 | 模型 | Attention 分类 | 实际模型入口或实现 | Ascend + Mooncake L3 判定 |
| --- | --- | --- | --- | --- |
| 1 | Qwen3-8B | 全量 GQA | `Qwen3ForCausalLM` / `Qwen3Attention` | 代码路径支持，见 A |
| 2 | Qwen3-32B | 全量 GQA | 同一 `Qwen3Attention` 实现，配置规模不同 | 代码路径支持，见 A |
| 3 | Qwen3-30B-A3B | 全量 GQA + MoE | `Qwen3MoeForCausalLM` / `Qwen3MoeAttention` | 代码路径支持，见 A |
| 4 | Qwen3-235B-A22B | 全量 GQA + MoE | 同一 `Qwen3MoeAttention` 实现 | 代码路径支持，见 A |
| 5 | MiniMax-M2.5 | 全量 GQA + MoE | `MiniMaxM2ForCausalLM` / `MiniMaxM2Attention` | 代码路径支持，见 A |
| 6 | Hy3 | 全量 GQA + MoE | `HYV3ForCausalLM` / `HYV3Attention` | 代码路径支持，见 A |
| 7 | DeepSeek-R1 | MLA + MoE | DeepSeek V3/V2 模型实现、`DeepseekV2AttentionMLA` | 代码路径支持，使用 page-first split-KV，见 B |
| 8 | Kimi-K2.6 | MLA + MoE | `KimiK25ForConditionalGeneration` 内嵌 `DeepseekV3ForCausalLM` | 语言模型 KV 路径支持，见 B |
| 9 | DeepSeek-V3.2 | DSA：MLA + 稀疏 indexer | `DeepseekV32ForCausalLM` / DeepSeek attention | KV 和 indexer 路径支持，见 C |
| 10 | GLM-5.1 | MLA + DSA | `GlmMoeDsaForCausalLM` | 代码路径支持，见 C |
| 11 | GLM-5.2 | MLA + DSA，包含 indexer 共享机制 | `GlmMoeDsaForCausalLM` | 代码路径支持，见 C |
| 12 | GLM-5.3 | MLA + DSA，包含 indexer 共享机制 | `GlmMoeDsaForCausalLM` | 代码路径支持，与 Flash 版不同，见 C |
| 13 | Qwen3-Next-80B-A3B-Instruct | GDN + full attention | `Qwen3NextForCausalLM` | KV + GDN 状态路径支持，见 D |
| 14 | Qwen3.5-397B-A17B | GDN + full attention + MoE | `Qwen3_5MoeForConditionalGeneration` | KV + GDN 状态路径支持，见 D |
| 15 | Qwen3.6-27B | GDN + full attention | `Qwen3_5ForConditionalGeneration` | 代码路径支持，见 D |
| 16 | Qwen3.6-35B-A3B | GDN + full attention + MoE | `Qwen3_5MoeForConditionalGeneration` | 代码路径支持，见 D |
| 17 | Qwen3.8-Max | GDN + full attention + MoE | `Qwen3_5MoeForCausalLM` | 代码路径支持，见 D |
| 18 | GLM-5.3-Flash | KDA + DSA | `Glm5NextForConditionalGeneration` | **存在 NPU 设备池 / host 布局 / 搬运接线缺口，见 E** |
| 19 | MiMo-V2-Flash | SWA GQA + full GQA，K/V 维度不对称 | `MiMoV2FlashForCausalLM` | **非对称 host pool 缺少 NPU 搬运分支，见 F** |
| 20 | DeepSeek-V4-Flash | CSA + HCA，配合 SWA 和压缩状态 | DeepSeek-V4 专用 attention 与缓存栈 | 专用多池路径支持，见 G |

多模态模型的上述判定针对语言模型前缀 KV / state；不能据此推断视觉编码器特征等其他数据也由同一 L3 路径缓存。

## A. 全量 GQA：6 个模型

### 模型确认

- **Qwen3-8B**：[`Qwen3Attention`](python/sglang/srt/models/qwen3.py#L193) 使用 `num_kv_heads` 和 `RadixAttention`。
- **Qwen3-32B**：复用上述实现，改变的是层数、head 数等模型配置。
- **Qwen3-30B-A3B**：[`Qwen3MoeAttention`](python/sglang/srt/models/qwen3_moe.py#L485) 使用 GQA；MoE 位于 FFN 路径。
- **Qwen3-235B-A22B**：复用同一个 `Qwen3MoeAttention`，不是因为参数规模增大就切换为 MLA。
- **MiniMax-M2.5**：[`MiniMaxM2Attention`](python/sglang/srt/models/minimax_m2.py#L632) 按 `num_key_value_heads` 构造 attention。
- **Hy3**：[`HYV3Attention`](python/sglang/srt/models/hunyuan_v3.py#L221) 使用独立 Q/K/V head 数及 `RadixAttention`。

### 缓存路径

```text
NPUMHATokenToKVPool
  -> MHATokenToKVPoolHost
  -> kernel_ascend
  -> Mooncake K/V 对象
```

GQA 在缓存实现中使用 MHA 类池，不能只根据类名将其误分类为模型层面的普通 MHA。

- Ascend 常规模型的池选择见 [`KVCacheConfigurator`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L1841)。
- Host 加载和备份具有 `kernel_ascend` 分支，见 [`load_to_device_per_layer`](python/sglang/srt/mem_cache/pool_host/mha.py#L566) 和 [`backup_from_device_all_layer`](python/sglang/srt/mem_cache/pool_host/mha.py#L729)。
- NPU 参数处理将此类模型的 host layout 设为 `page_first_direct`，见 [`set_default_server_args`](python/sglang/srt/hardware_backend/npu/utils.py#L181)。
- Mooncake 注册 host buffers 并通过页元数据存取 K/V，见 [`register_mem_pool_host`](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L865)。

因此，以上 6 个模型具有共同的、已接通的常规 GQA 缓存路径。该判断不自动覆盖特殊 KV 量化布局。

## B. 普通 MLA：DeepSeek-R1、Kimi-K2.6

### 模型确认

- **DeepSeek-R1**：使用 DeepSeek V3/V2 模型实现中的 [`DeepseekV2AttentionMLA`](python/sglang/srt/models/deepseek_v2.py#L2063)。
- **Kimi-K2.6**：本地 [`KimiK25ForConditionalGeneration`](python/sglang/srt/models/kimi_k25.py#L690) 实例化 `DeepseekV3ForCausalLM`；[官方配置](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json) 与此一致。它不是 KDA 模型。

### 缓存路径

```text
NPUMLATokenToKVPool
  -> MLATokenToKVPoolHost / page_first_kv_split
  -> kernel_ascend
  -> Mooncake 单个逻辑页对应的多个物理 buffers
```

- NPU MLA 设备池构建见 [`_build_ascend_mla_kv_pool`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L2290)。
- NPU host 加载支持 `page_first_kv_split`，见 [`MLATokenToKVPoolHost`](python/sglang/srt/mem_cache/pool_host/mla.py#L661)。
- [`get_hybrid_pool_buffer`](python/sglang/srt/mem_cache/pool_host/mla.py#L1089) 导出实际物理 buffers；[`get_page_buffer_meta`](python/sglang/srt/mem_cache/pool_host/mla.py#L1109) 提供一页中各 buffer 的指针和大小。
- 普通 split MLA 路径保存 latent KV 和所需 RoPE 数据，不能只注册一个主 buffer。

两个模型的主模型 KV 路径支持；DCP、draft pool 和特殊 KV 格式见后文限制。

## C. DSA：DeepSeek-V3.2、GLM-5.1、GLM-5.2、GLM-5.3

### 模型确认

- **DeepSeek-V3.2**：[`DeepseekV32ForCausalLM`](python/sglang/srt/models/deepseek_v2.py#L3633) 进入 DeepSeek attention 路径，DSA 配置带来额外 indexer 状态。
- **GLM-5.1**：[`GlmMoeDsaForCausalLM`](python/sglang/srt/models/glm4_moe.py#L1288) 继承 `DeepseekV2ForCausalLM`。
- **GLM-5.2**：同一 DSA 入口，配置包含 indexer 共享机制。
- **GLM-5.3**：[官方配置](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json) 仍是 `GlmMoeDsaForCausalLM`，包含 `index_topk`、`index_topk_freq`、`indexer_types`；不能与 GLM-5.3-Flash 混为一类。

### 必须保存的状态

DSA 的支持判断必须同时检查主 KV 和 indexer：

```text
MLA KV / RoPE（或相应 packed 表示）
  + index_k
  + index_k_scale（如果设备格式需要）
```

Ascend DSA 使用 NPU MLA 设备池，构建时传入 indexer 信息，见 [`_build_ascend_mla_kv_pool`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L2290)。Host 的 [`get_hybrid_pool_buffer`](python/sglang/srt/mem_cache/pool_host/mla.py#L1100) 明确导出 `index_k_buffer` 和 `index_k_scale_buffer`；页元数据按照实际 indexer 层数、宽度和 dtype 计算，见 [`get_page_buffer_meta`](python/sglang/srt/mem_cache/pool_host/mla.py#L1133)。

本地代码对启用 NPU HiCache 时的 index-K 省略也有专门处理，见 [`_should_elide_dsa_index_k`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L137)。不能把非 HiCache 模式下的紧凑 indexer 分配直接套入当前 host layout。

因此，以上 4 个模型的常规 DSA 路径包含必要 indexer 数据，可以判为代码路径支持；这不是从普通 MLA 支持外推得到的结论。

## D. GDN 混合：5 个 Qwen 模型

### 模型确认

- **Qwen3-Next-80B-A3B-Instruct**：[`Qwen3NextModel`](python/sglang/srt/models/qwen3_next.py#L1044) 按层类型选择 Gated DeltaNet 或 attention。
- **Qwen3.5-397B-A17B**：[`qwen3_5.py`](python/sglang/srt/models/qwen3_5.py#L1660) 分派 linear-attention 和 attention decoder layer。
- **Qwen3.6-27B**：[官方配置](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json) 为 `Qwen3_5ForConditionalGeneration`，复用 Qwen3.5 dense 实现。
- **Qwen3.6-35B-A3B**：[官方配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json) 为 `Qwen3_5MoeForConditionalGeneration`。
- **Qwen3.8-Max**：教程对应的 [Qwen3.8-2.4T-A95B 配置](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json) 为 `Qwen3_5MoeForCausalLM`，仍是 GDN/full-attention 混合结构。

### 缓存路径

线性 attention 层缓存的是可恢复的状态检查点，而不是普通逐 token K/V：

```text
full-attention K/V -> MHA host pool
GDN recurrent state + convolution state -> MambaPoolHost
  -> HostPoolGroup
  -> Mooncake 多池对象
```

- [`build_hybrid_mamba_stack`](python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1187) 将 KV 和 Mamba state pool 组合进缓存栈。
- [`MambaPoolHost`](python/sglang/srt/mem_cache/pool_host/mamba.py#L537) 实现 NPU H2D 加载；[D2H 分支](python/sglang/srt/mem_cache/pool_host/mamba.py#L670) 实现状态备份。
- [`get_page_buffer_meta`](python/sglang/srt/mem_cache/pool_host/mamba.py#L903) 导出 temporal 和 conv buffers。
- Mooncake 为 temporal 和每个 conv state 生成对应对象键，见 [`_get_hybrid_page_component_keys`](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1070)，并通过 [`batch_get_v2` / `batch_set_v2`](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1435) 存取。

以上 5 个模型不能因为含线性 attention 就判为不支持；当前代码确实具备 KV 与 GDN 状态的完整接入路径。不过，缓存命中受状态检查点边界约束，不能理解为任意 token 边界都能像纯 full-attention 一样复用。

## E. GLM-5.3-Flash：KDA + DSA 的 NPU 路径未接通

[官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json) 指定 `Glm5NextForConditionalGeneration`，包含 KDA 线性层和 DSA 层，并设置 `index_kpool=4`、`index_kpool_compress=true`。
本地 [`Glm5NextDecoderLayer`](python/sglang/srt/models/glm5_next.py#L777) 明确分派 `Glm5NextLinearAttention` 和 `DeepseekV2AttentionMLA`。

当前默认代码路径存在以下不匹配：

1. [`_hybrid_full_attention_pool_class`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L2848) 先判断 MLA + DSA 并返回 `dsa_pool_class`，此分支早于 NPU 专用池选择。
2. 默认 DSA 类来自 [`DSATokenToKVPool`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L1814)，混合池据此构建设备存储，见 [`HybridLinearKVPool`](python/sglang/srt/mem_cache/memory_pool.py#L4335)。
3. 该类复用普通 MLA buffer 描述，提供 tensor tuple，见 [`_mla_buffer_info`](python/sglang/srt/mem_cache/memory_pool.py#L4936)。
4. NPU HiCache 参数则选择 `kernel_ascend` 和 split-KV layout，见 [`NPU 参数处理`](python/sglang/srt/hardware_backend/npu/utils.py#L181)。
5. Host 只有收到 `MLASplitBuffers` 才建立 `_native_buffers`，见 [`MLATokenToKVPoolHost`](python/sglang/srt/mem_cache/pool_host/mla.py#L205)；split-KV 初始化却直接读取 `native.index_layer_indices`，见 [同文件初始化分支](python/sglang/srt/mem_cache/pool_host/mla.py#L396)。因此普通 tuple 描述不能满足该路径。
6. 独立 [`DSAIndexerPoolHost`](python/sglang/srt/mem_cache/pool_host/dsa.py#L310) 也没有接通对应的 `kernel_ascend` 搬运，不能只修改 MLA host layout 就认为修复完成。

**判定：当前默认 Ascend HiCache + Mooncake L3 不能标为支持。**
KDA 状态的通用 host 实现存在，但不代表 DSA + KDA 整体已经接通。修复需要同时覆盖设备池描述、NPU 搬运、压缩 indexer 及 KDA 状态的恢复语义，然后验证真实 L3 命中后的输出一致性。

## F. MiMo-V2-Flash：非对称 KV host pool 缺少 NPU 搬运

模型按层选择 SWA 或 full attention，见 [`MiMoV2DecoderLayer`](python/sglang/srt/models/mimo_v2.py#L872)。[官方配置](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/raw/main/config.json) 中：

```text
full attention: head_dim=192, v_head_dim=128
SWA attention:  swa_head_dim=192, swa_v_head_dim=128
```

阻断链路为：

```text
K/V head dimension 不相等
  -> AsymmetricMHATokenToKVPoolHost
  -> NPU 强制选择 kernel_ascend
  -> 非对称 host pool 没有该搬运分支
```

- NPU SWA 设备池保留 K/V 各自维度，见 [`_build_ascend_swa_kv_pool`](python/sglang/srt/mem_cache/kv_cache_configurator.py#L2219)。
- NPU buffer 描述也保留独立的 `head_dim` / `v_head_dim`，见 [`_mha_buffer_info`](python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L236)。
- [`get_mha_host_pool_cls`](python/sglang/srt/mem_cache/pool_host/mha.py#L1751) 因此选择 `AsymmetricMHATokenToKVPoolHost`。
- 此类的 H2D 和 D2H 实现只处理 `kernel`、`direct`，其余值抛出 `Unsupported IO backend`，见 [加载路径](python/sglang/srt/mem_cache/pool_host/mha.py#L1511) 与 [备份路径](python/sglang/srt/mem_cache/pool_host/mha.py#L1582)。

**判定：当前默认 Ascend HiCache + Mooncake L3 路径不支持。**
SWA 多池管理和 Mooncake 页存取能力本身存在，具体缺口是 NPU/CPU 的非对称 K/V 搬运。修复应为 K、V 分别使用正确的 stride 和复制大小，并同时验证 full 与 SWA 两个池，不能复用一个共享 head dimension。

## G. DeepSeek-V4-Flash：专用压缩多池路径

模型使用 CSA/HCA 压缩稀疏 attention，代码按 compression ratio 构建相关模块，见 [`MqaAttentionBase`](python/sglang/srt/models/deepseek_v4.py#L1010)。不能按普通 MLA 或普通 SWA 的单池缓存分析。

支持依据：

- [`build_deepseek_v4_hicache_stack`](python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L788) 建立 SWA、C4/C128、indexer 和压缩状态等 host pools。
- NPU 的 C128 由独立树组件管理，见 [`_DeepSeekV4Strategy.build`](python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L2068)。
- [`DeepSeekV4PagedHostPool`](python/sglang/srt/mem_cache/memory_pool_host.py#L585) 与 [`DeepSeekV4StateHostPool`](python/sglang/srt/mem_cache/memory_pool_host.py#L1099) 均有 `kernel_ascend` 搬运实现。
- 两类 host pool 均导出 L3 页元数据，见 [paged 元数据](python/sglang/srt/mem_cache/memory_pool_host.py#L776) 和 [state 元数据](python/sglang/srt/mem_cache/memory_pool_host.py#L1286)。
- Mooncake [`register_mem_host_pool_v2`](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L881) 注册额外池，v2 batch I/O 承接多池读写。

**判定：专用多池代码路径支持。** 必须通过整套 V4 stack 保存和恢复所需状态，不能只备份主 KV 或只依据某一个池的支持情况判断。

## 使用条件与明确限制

| 项目 | 当前代码结论 | 依据 |
| --- | --- | --- |
| NPU I/O backend | 启用 HiCache 后，NPU 参数处理选择 `kernel_ascend` | [`npu/utils.py`](python/sglang/srt/hardware_backend/npu/utils.py#L181) |
| GQA / GDN host layout | NPU 常规路径选择 `page_first_direct` | 同上 |
| MLA / DSA L3 host layout | 使用 `page_first_kv_split`；可显式请求 `page_first`，由 NPU 参数处理转成 split 版本 | 同上及 [`mla.py`](python/sglang/srt/mem_cache/pool_host/mla.py#L1109) |
| MLA layer-first | NPU 可生成 `layer_first_kv_split`，但 L3 页元数据接口没有该布局分支；不能仅打开 Mooncake 而不检查最终布局 | [`mla.py`](python/sglang/srt/mem_cache/pool_host/mla.py#L1109) |
| V4 host layout | 有专用布局处理，可选择 `page_first`；不使用普通 MLA 的 split 布局 | [`npu/utils.py`](python/sglang/srt/hardware_backend/npu/utils.py#L198) |
| DSA + DCP + HiCache | 参数阶段显式拒绝 | [`_dsa_dcp_overrides`](python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py#L18) |
| NPU MLA split-KV + DCP L3 | 页元数据接口显式抛出 `NotImplementedError` | [`mla.py`](python/sglang/srt/mem_cache/pool_host/mla.py#L1111) |
| MTP / speculative decoding | 需单独检查 draft 缓存；某些 NPU MLA draft indexer 明确不备份，不能将主模型支持等同于全部 draft 状态支持 | [`mla.py`](python/sglang/srt/mem_cache/pool_host/mla.py#L835) |
| INT8 Mamba checkpoint | 与 HiCache 显式互斥 | [`handle_int8_mamba_checkpoint`](python/sglang/srt/arg_groups/mamba_hook.py#L74) |
| Mamba slot side states | 若额外注册了 slot side states，当前 zero-copy 页元数据接口拒绝；不能把常规 GDN 结论扩展到任意扩展状态 | [`mamba.py`](python/sglang/srt/mem_cache/pool_host/mamba.py#L903) |
| 禁用 radix cache | `--disable-radix-cache` 与 HiCache 互斥 | [`kv_cache_hook.py`](python/sglang/srt/arg_groups/kv_cache_hook.py#L392) |
| unified L3 key scheme | 当前只允许 file backend，不能用于 Mooncake | [`resolve_hicache_key_scheme`](python/sglang/srt/arg_groups/hicache_hook.py) |

W8A8/W4A8 等权重量化不等于 KV cache 量化。特殊 KV dtype、scale buffer、统一内存和其他布局组合须独立确认。教程中的镜像版本也不一定与本报告的代码提交一致，不能将此审查结论无条件外推到旧版本。

## 已有测试证据与剩余验证

| 测试 | 可以支持的判断 | 不能据此宣称的结论 |
| --- | --- | --- |
| [`test_npu_hierarchical_cache.py`](test/registered/npu/basic_function/HiCache/test_npu_hierarchical_cache.py) | 存在 Qwen3-8B NPU HiCache 测试 | 不能证明所有 GQA 型号均通过 Mooncake L3 |
| [`test_npu_hierarchical_cache_ttft_mha.py`](test/registered/npu/basic_function/HiCache/test_npu_hierarchical_cache_ttft_mha.py) | 存在 Qwen3-32B NPU HiCache 测试 | 不能等同于该型号所有量化、并行组合的 L3 验证 |
| [`test_npu_hierarchical_cache_mla.py`](test/registered/npu/basic_function/HiCache/test_npu_hierarchical_cache_mla.py) | 存在 DeepSeek-R1 NPU HiCache 测试 | 不能直接覆盖 DSA、MTP indexer 或 Mooncake L3 |
| [`test_npu_hicache_mamba.py`](test/registered/npu/basic_function/HiCache/test_npu_hicache_mamba.py) | Qwen3.5-0.8B + NPU + file backend 的 L3 状态恢复测试 | 不是 5 个目标 GDN 型号各自的 Mooncake 端到端测试 |
| [`test_npu_hicache_sparse_dispatch.py`](test/registered/unit/mem_cache/test_npu_hicache_sparse_dispatch.py) | NPU MLA / sparse host 搬运分派与 buffer 绑定测试 | Mock / 单元测试不代表真实设备及网络传输成功 |
| [`test_dsv4_hicache_l2.py`](test/registered/unit/mem_cache/test_dsv4_hicache_l2.py) | V4 host pool 和 NPU 布局相关单元覆盖 | 不是完整模型端到端性能与正确性验证 |
| [`test_dsv4_hicache_l3.py`](test/registered/unit/mem_cache/test_dsv4_hicache_l3.py) | C128 L3 边界、恢复及清理相关覆盖 | 不是 V4-Flash + NPU + Mooncake 的实机验收 |
| [`test_mooncake_group_semantics.py`](test/registered/unit/mem_cache/test_mooncake_group_semantics.py) | Mooncake 多对象、多池及 C128 边界语义覆盖 | 不能替代各型号真实 L3 命中后的输出一致性检查 |

实机验收应至少验证：写入完整前缀及必要附属状态、移除或驱逐 L1/L2 数据、确认再次请求真正产生 L3 storage hit，并对比无缓存重算的输出。混合模型还应覆盖状态检查点边界；DSA 覆盖 indexer 恢复；V4 覆盖 C128 完整块边界及部分读取失败。当前报告没有声称完成这些实机步骤。
