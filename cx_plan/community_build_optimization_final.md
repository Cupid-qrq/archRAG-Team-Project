# 社区构建算法优化说明

## 目标

- 优化 ArchRAG 索引阶段的社区构建算法.
- 在不改变整体 ArchRAG 查询链路和 HCHNSW 索引接口的前提下, 改善社区划分质量.
- 保留原有 `weighted_leiden` 等聚类方式, 通过新增参数启用优化路径.

## 修改文件

- `src/utils.py`
- `src/attr_cluster.py`
- `src/community_report.py`
- `src/lm_emb.py`

## 新增接口参数

### `--cluster_method mutual_knn_leiden`

启用新的社区构建路径.

流程:

1. 使用实体 embedding 构建 Mutual KNN 语义图.
2. 保留原始 KG 边, 并用语义相似度更新边权.
3. 在多个候选 `max_cluster_size` 上运行 Leiden.
4. 使用综合评分选择最优划分.

### `--mutual_knn_k`

Mutual KNN 图构建时每个节点保留的 top-k 近邻数量.

默认值:

```text
8
```

### `--mutual_knn_min_sim`

新增语义边的最小余弦相似度阈值.

默认值:

```text
0.0
```

### `--mutual_knn_original_edge_weight`

当原始 KG 边的语义相似度低于 `mutual_knn_min_sim` 时使用的保底边权.

默认值:

```text
0.05
```

### `--dynamic_cluster_sizes`

动态 Leiden 搜索的候选社区大小列表.

默认值:

```text
8,12,16,24,32
```

## `src/utils.py` 修改

### 新增 `parse_dynamic_cluster_sizes`

将命令行传入的字符串解析成有序, 去重, 合法的整数列表.

约束:

- 候选值必须大于 1.
- 空列表直接报错.

### 新增 `_normalized_node_embeddings`

统一读取图节点 embedding, 构建归一化矩阵.

作用:

- 后续使用内积等价计算 cosine similarity.
- 避免每次建边重复归一化.

### 新增 `compute_mutual_knn_graph`

基于实体 embedding 构建 Mutual KNN 语义增强图.

核心逻辑:

- 使用 FAISS `IndexFlatIP` 搜索 top-k 语义近邻.
- 只保留互为近邻的语义边, 降低单向近邻带来的噪声.
- 原始 KG 边仍然保留, 但边权改为语义相似度.
- 对相似度过低的原始边使用 `mutual_knn_original_edge_weight`.

输出:

- 带 `weight` 和 `edge_type` 的 NetworkX 图.
- 日志打印节点数, 原始边数, 新增语义边数, 总边数和参数.

## `src/attr_cluster.py` 修改

### 新增 `_final_leiden_mapping`

封装 Leiden 结果读取逻辑, 只保留最终社区.

返回:

- `c_n_mapping`: community -> node list.
- `node_cluster`: node -> community.

### 新增 `_semantic_cohesion`

计算社区内部实体 embedding 的语义凝聚度.

用途:

- 避免只依赖图结构 modularity.
- 鼓励同一社区内实体语义更接近.

### 新增 `_conductance_penalty`

计算跨社区边权占比.

用途:

- 惩罚大量强边被切开的划分.

### 新增 `_partition_modularity`

计算 NetworkX modularity.

用途:

- 保留原始图聚类的结构质量指标.

### 新增 `_score_partition`

综合评分公式:

```text
score = modularity
      + semantic_cohesion
      - conductance_penalty
      - singleton_ratio
      - 0.5 * size_imbalance
      - oversized_penalty
```

评分目标:

- 奖励结构模块度.
- 奖励社区语义凝聚度.
- 惩罚跨社区强连接.
- 惩罚过多单节点社区.
- 惩罚社区大小严重不均衡.
- 惩罚超过目标大小的社区.

### 新增 `_is_effective_partition`

过滤无效划分.

拒绝条件:

- 空划分.
- 每个节点几乎都独立成社区.
- 单节点社区比例过高.

### 新增 `compute_dynamic_leiden`

对 `dynamic_cluster_sizes` 中的每个候选大小运行 Leiden, 计算评分, 选择最高分的有效划分.

日志会输出:

- 每个候选 `max_cluster_size`.
- 对应综合评分.
- modularity, semantic_cohesion, conductance_penalty 等指标.
- 最终选中的候选大小.

### 修改 `attr_cluster`

新增分支:

```text
cluster_method == "mutual_knn_leiden"
```

该分支使用:

1. `compute_mutual_knn_graph`
2. `compute_dynamic_leiden`

其他原有聚类路径保持不变.

## `src/community_report.py` 修改

新增 `_validate_report_embeddings`.

目的:

- 防止 embedding 服务失败后产生 `None` 或 `NaN`.
- 防止错误 embedding 被写入 `tmp_community_df_*.csv`.
- 防止后续 HCHNSW 拼接时报维度错误.

校验条件:

- embedding 必须是一维向量.
- 默认维度必须是 1024.
- 所有值必须是有限数值.

## `src/lm_emb.py` 修改

修改 `openai_embedding` 的异常处理.

之前行为:

```text
失败后打印错误并返回 None
```

现在行为:

```text
失败后直接抛出 RuntimeError
```

原因:

- 返回 `None` 会被 pandas 写成 `NaN`.
- 后续缓存会污染索引构建.
- 直接失败能定位真正的 embedding 服务问题.

## Smoke 测试结果

使用小输入完成 GraphRAG -> ArchRAG index -> query 全链路测试.

Index 结果:

```text
finish compute HCa RAG index
Finished computing HCa RAG index in 130.16 seconds ()
Create Index Total Token Usage: 31688
```

Query 结果:

```text
Number of questions: 8
Finish query Time: 77.01 seconds
Hit: 87.5000
Precision: 80.3686
Recall: 96.8750
F1: 83.4918
EM: 75.0000
```

## 结论

- 新增 `mutual_knn_leiden` 路径后, 社区构建可以同时利用 KG 原始结构和实体语义相似度.
- 动态候选搜索避免固定 `max_cluster_size` 带来的不稳定.
- embedding 失败校验修复了 `NaN` 缓存导致的 HCHNSW 维度错误.
- smoke 测试已证明优化代码可以完成端到端索引和查询流程.
