# KernelWiki

以 **operator 为核心的 GPU kernel knowledge base**：记录不同 backend 的实现、适用条件、性能证据和优化经验，帮助开发者与 GEAK 编写、选择和改进 kernel。

## 两类内容

| 内容 | 入口 | 用途 |
| --- | --- | --- |
| 文本 knowledge | [Operator](knowledge/operator/README.md) · [General](knowledge/general/README.md) | 实现原理、源码入口、效果与适用范围、调优方法、历史经验 |
| Kernel patch | [写入与检索接口](kernel_patch/README.md) | 通过现有 GEAK KB 服务保存、检索和下载 patch / tuning artifact |

```text
knowledge/
  operator/<operator>/
    README.md
    backend/<backend>_aiter.md
    backend/<backend>_notes.md
    learned_knowledge/
  general/
    profile_bottleneck/
    optimization/
    backend/
    hardware/
    quantization/
    ...
kernel_patch/
  client.py
```

## 从哪里开始

- [Dense GEMM](knowledge/operator/dense_gemm/README.md) · [量化 GEMM](knowledge/operator/scaled_quant_gemm/README.md) · [Grouped GEMM / MoE](knowledge/operator/grouped_gemm_moe/README.md)
- [MLA](knowledge/operator/mla_attention/README.md) · [Paged decode attention](knowledge/operator/attention_decode_paged/README.md) · [RMSNorm](knowledge/operator/rmsnorm/README.md)
- [瓶颈分析](knowledge/general/profile_bottleneck/README.md) · [优化方法](knowledge/general/optimization/README.md) · [知识写作与证据约定](knowledge/README.md)

## 首版内容

- GEAK 来源：[`02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/tree/02bfbb0b72746941be3ab3ad48fb21a284ae2fde)。迁入 894 篇原有 Markdown，包含 166 张 kernel/e2e 历史经验卡片，并保留相关 recipe 资源。
- AITER 来源：[`e9995854abbd`](https://github.com/ROCm/aiter/tree/e9995854abbd935f0190b19abebe9938b6b90cca)。增加 159 篇 operator/backend 实现参考，关联 919 个实现及辅助源码文件，其中 824 个在近三个月有修改；另按家族引用 3271 个 ASM/HSA 产物与元数据文件。
- 当前共 63 个 operator 目录。源码参考覆盖与实测性能覆盖分开：本次没有新增 GPU benchmark，也没有把历史 SOTA 标记升级为当前最快结论。
- GEAK 已消耗的 inbox、旧生成索引和索引生成脚本未作为正式知识迁入；来源仍可在固定版本的 GEAK 中查阅。

AITER 是实现知识的主要来源。文本中保留固定版本的源码、配置和测试链接；整理重点是 kernel 本身。不同 backend 的“最好实现”需要结合 GPU、dtype、shape/layout 和可比较的测量判断。

## 使用 patch 接口

```bash
python -m pip install -e .
export GEAK_KB_URL="https://your-existing-kb-service"
# 使用现有 GEAK_KB_STORE_TOKEN / KB_STORE_TOKEN 配置
```

```python
from kernel_patch import PatchClient, kernel_addresses

client = PatchClient.from_env()
addresses = kernel_addresses(
    kernel_name="_rms_norm_kernel", backend="triton", gfx="gfx950", rocm_version="7.3"
)
result = client.search(canonical_ids=addresses, top_k=5)
```

接口兼容 `GEAK_KB_STORE_URL`、`KB_STORE_URL` 和已有 canonical IDs。详细写入、检索范围、错误处理和下载示例见 [kernel_patch](kernel_patch/README.md)。文本直接从本仓库的 Markdown 读取；Python 包提供 patch 访问能力。

## 开发检查

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## 来源与许可

GEAK 迁移内容保留出处与 Apache-2.0 许可；AITER 引用及摘录保留其 MIT 许可说明。见 [LICENSE](LICENSE.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
