# ArchRAG 三元组-源文本映射优化说明

本文记录本轮在当前 `main` 分支上新增的三元组-源文本映射优化。该优化不依赖未合并的 `mutual_knn_leiden` 分支。

## 1. 优化目标

原 ArchRAG 离线阶段会为社区调用 LLM 生成报告，在线阶段主要把社区摘要交给最终生成器。现在新增一层 evidence provenance：

```text
relationship.text_unit_ids -> create_final_text_units.id -> source text
```

这样社区和关系都可以直接携带原始文本证据。大社区可以用源文本构造 extractive community record，减少社区报告 LLM 调用；在线检索也能把证据文本放进 prompt，提高可解释性。

## 2. 新增文件

### `src/triple_text_mapping.py`

核心职责：

- 读取 `create_final_text_units.parquet/csv`。
- 解析 `relationship_df["text_unit_ids"]`，兼容 list、字符串 list、numpy/Arrow 数组。
- 构建 `triple_text_mapping`：

```python
{
    relationship_id: {
        "description": "...",
        "head": head_id,
        "tail": tail_id,
        "source_chunks": [text_unit_id],
        "chunk_texts": [source_text],
        ...
    }
}
```

- 计算 `chunk_weights`，即每个 text unit 被多少条关系引用。
- 保存和读取：
  - `triple_text_mapping.pkl`
  - `chunk_weights.pkl`
- 为在线检索提供：
  - `enrich_community_with_source_text()`
  - `enrich_relationships_with_source_text()`
  - `make_extractive_community_report()`

## 3. 离线索引改动

### Artifact 复制

`copy_latest_archrag_artifacts.sh` 现在会同时复制：

```text
create_final_entities.parquet
create_final_relationships.parquet
create_final_text_units.parquet
```

如果最新 GraphRAG 输出缺少 text units，脚本会直接提示缺失文件，避免后续索引阶段静默降级。

### `src/index.py`

在 `read_graph_nx()` 后新增映射构建：

1. 读取 `create_final_text_units.parquet`。
2. 用关系表中的 `text_unit_ids` 精确 join 到源文本。
3. 保存 `triple_text_mapping.pkl` 和 `chunk_weights.pkl`。
4. 把 mapping 传入 `attr_cluster()`。
5. 社区构建完成后输出 `community_source_text.csv`，便于人工检查每个社区对应的证据文本。

### `src/community_report.py`

社区报告现在支持三种模式：

```text
llm        全部社区仍调用 LLM
extractive 全部可映射社区直接用源文本
hybrid     默认模式，大社区用源文本，小社区仍调用 LLM
```

hybrid 的判断逻辑：

- 如果社区相关关系数量 `>= --extractive_large_community_threshold`，且能找到源文本，则生成 direct_text 社区记录。
- 否则保留原 LLM 社区报告逻辑。

direct_text 社区仍保留原字段：

```text
title
summary
findings
rating
rating_explanation
community_text
embedding
```

因此不会破坏 HCHNSW 构建。

## 4. 在线检索改动

### `src/inference.py`

`load_index()` 会尝试读取：

```text
triple_text_mapping.pkl
chunk_weights.pkl
```

如果旧索引目录没有这两个文件，会打印提示并继续运行。

`hcarag_retrieval()` 会给检索结果补充：

```text
source_texts
source_text_unit_ids
source_relationship_ids
```

补充对象包括：

- `topk_community`
- `topk_related_r`
- HyperNode 命中的关系候选

### `src/client_reasoning.py`

`prep_community_content()` 现在优先使用 `source_texts` 作为社区内容。如果没有源文本证据，则回退到原来的 `summary`。

这意味着即使 `topk_chunk=0`，系统也可以通过三元组映射把源文本放进最终生成 prompt。

## 5. 新增参数

离线索引参数：

```text
--text_unit_filename create_final_text_units.parquet
--enable_triple_text_mapping True
--community_report_mode hybrid
--extractive_large_community_threshold 5
--source_text_top_k 3
--source_text_max_tokens 0
```

在线检索参数：

```text
--enable_triple_text_mapping True
--source_text_top_k 3
--source_text_max_tokens 0
```

`source_text_max_tokens=0` 表示沿用 `max_tokens` 做内部裁剪。

`dataset/index.sh` 和 `dataset/query.sh` 已显式设置：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

因此在新的 WSL shell 中可以直接运行脚本，不需要手动导出项目路径。

## 6. 运行方式

推荐流程：

```bash
source /home/qrq/miniconda3/etc/profile.d/conda.sh
conda activate archrag

bash copy_latest_archrag_artifacts.sh corpus/output archrag
bash dataset/index.sh
bash dataset/query.sh
```

索引完成后检查：

```text
archrag_index/triple_text_mapping.pkl
archrag_index/chunk_weights.pkl
archrag_index/community_source_text.csv
```

日志中会打印：

```text
Built triple-text mapping: ...
Community report source types: {'direct_text': ..., 'llm_report': ...}
```

## 7. 验证结果

已在 WSL conda 环境 `/home/qrq/miniconda3/envs/archrag` 下完成静态验证：

- `py_compile` 通过：
  - `src/triple_text_mapping.py`
  - `src/index.py`
  - `src/attr_cluster.py`
  - `src/community_report.py`
  - `src/client_reasoning.py`
  - `src/inference.py`
  - `src/utils.py`
- 使用 `corpus/output/20260520-214913/artifacts` 构建映射：
  - 139 条关系全部映射到源文本。
  - 12 个 text units 生成了 chunk 权重。
- 使用旧索引目录 `expanded_archrag_real_index` 测试：
  - 缺少 mapping 文件时 `load_index()` 正常回退，不影响旧索引查询。

## 8. 后续注意事项

- 当前 `archrag/` 目录里旧数据可能还没有 `create_final_text_units.parquet`。重新运行 `copy_latest_archrag_artifacts.sh` 后即可补齐。
- 如果需要完全复现实验，请用 `--community_report_mode llm` 关闭 extractive 社区，作为原始基线。
- 如果想最大化离线省 token，可用 `--community_report_mode extractive`，但抽象问题可能会损失概括能力。
- 当前实现以 GraphRAG 的 `text_unit_ids` 为唯一可信 provenance，不使用关系描述的模糊匹配。
