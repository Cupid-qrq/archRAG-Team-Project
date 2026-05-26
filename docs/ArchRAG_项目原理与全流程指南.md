# ArchRAG 项目原理与全流程指南

这份文档面向刚接触 GraphRAG、ArchRAG 和这份仓库的读者。它会先用直觉解释项目解决什么问题，再逐步说明 GraphRAG、知识图谱、社区、层次索引、HCHNSW、在线检索和答案生成的关系，最后给出本仓库当前扩展示例语料的运行路径。

## 1. 这个项目到底在做什么

普通 RAG 的基本流程是：

1. 把文档切成片段。
2. 给每个片段做 embedding。
3. 用户提问时，用问题 embedding 去向量库里找相似片段。
4. 把相似片段塞进 LLM，让 LLM 回答。

这个方法简单，但有几个明显问题：

- 如果问题需要多跳推理，答案可能分散在多个片段里，单纯向量相似度不一定能找全。
- 如果问题比较抽象，比如“这个项目的主要风险是什么”，单个片段往往太细，缺少高层总结。
- 如果直接塞很多片段给 LLM，token 成本高，而且长上下文里容易丢重点。

ArchRAG 的思路是：不要只把文档当作一堆孤立文本片段，而是先从文档里抽出“实体”和“关系”，形成知识图谱，再把图里的实体组织成多层社区。这样查询时可以同时拿到：

- 实体级细节：适合回答具体事实问题。
- 低层社区摘要：适合回答局部主题问题。
- 高层社区摘要：适合回答全局、抽象、综合问题。

一句话概括：

> ArchRAG 是一个把 GraphRAG 生成的知识图谱进一步做“属性增强层级社区索引”的图 RAG 系统。

## 2. 本仓库的主要模块

仓库可以分成四块：

| 路径 | 作用 |
| --- | --- |
| `src/graphrag/` | 改造/内置的 Microsoft GraphRAG，用于从文档抽取实体、关系、文本块等基础图谱数据 |
| `src/index.py` | ArchRAG 离线建索引入口 |
| `src/inference.py` | ArchRAG 在线检索和生成入口 |
| `src/attr_cluster.py` | 属性增强层级社区聚类 |
| `src/hchnsw_index.py` | 构建和读取自定义 HCHNSW/Faiss 索引 |
| `src/community_report.py` | 用 LLM 为每个社区生成摘要 |
| `src/client_reasoning.py` | 层级摘要、问题层级判断、map-reduce 生成 |
| `HCHNSW/faiss/` | 修改过的 Faiss，新增 `IndexHCHNSWFlat` 等能力 |
| `corpus/` | 当前示例语料、GraphRAG 配置和输出 |
| `expanded_archrag_real_index/` | 本轮扩展语料的 ArchRAG 索引输出 |

## 3. GraphRAG 是什么

GraphRAG 是 Microsoft 提出的图增强 RAG 框架。它的核心思想是：先把原始文档转换成知识图谱，再基于图谱做检索和总结。

### 3.1 GraphRAG 的输入

输入是普通文本文件，例如本仓库现在的：

```text
corpus/input/doc_0001_overview.txt
corpus/input/doc_0002_energy.txt
...
corpus/input/doc_0012_risks_next_steps.txt
```

这些文档描述了一个虚构的 Aurora Harbor 城市韧性项目，包含交通、能源、水务、医院、避难所、数据治理、网络安全等互相关联的信息。

### 3.2 GraphRAG 的处理步骤

GraphRAG 大致做这些事：

1. 文档加载  
   从 `corpus/input/` 读取文本。

2. 文本切块  
   根据 `corpus/settings.yaml` 里的 `chunks.size` 和 `chunks.overlap` 切成 text units。

3. 实体抽取  
   LLM 从每个 chunk 中抽出实体，例如：
   - `AURORA HARBOR RESILIENCE PROGRAM`
   - `METROLINK TRANSIT`
   - `HARBORGRID ENERGY`
   - `BAYVIEW HOSPITAL`
   - `ROUTE 7`

4. 关系抽取  
   LLM 抽出实体之间的关系，例如：
   - HarborGrid Energy builds microgrids for Bayview Hospital.
   - MetroLink Route 7 connects East Market Depot, Bayview Hospital, and North Pier Shelter.
   - Civic Data Trust maintains the Open Harbor Portal.

5. 实体合并与描述总结  
   同一个实体可能在多个 chunk 出现，GraphRAG 会合并实体，并生成统一描述。

6. 关系合并  
   多次出现的关系会被合并或加权。

7. 输出 parquet/csv  
   生成 ArchRAG 后续要用的结构化文件。

### 3.3 GraphRAG 的关键输出

本项目最关心两个文件：

```text
create_final_entities.parquet
create_final_relationships.parquet
```

其中 `create_final_entities.parquet` 通常包含：

| 字段 | 含义 |
| --- | --- |
| `id` | 内部 UUID |
| `name` | 实体名称 |
| `type` | 实体类型 |
| `description` | LLM 总结出的实体描述 |
| `human_readable_id` | 可读的整数 ID，后续图构建会使用 |
| `description_embedding` | 实体描述 embedding |
| `text_unit_ids` | 该实体来自哪些文本块 |

`create_final_relationships.parquet` 通常包含：

| 字段 | 含义 |
| --- | --- |
| `source` | 起点实体名 |
| `target` | 终点实体名 |
| `description` | 关系描述 |
| `weight` | 关系权重 |
| `source_degree` | 起点实体度数 |
| `target_degree` | 终点实体度数 |
| `rank` | 关系重要性相关分数 |
| `head_id` / `tail_id` | 映射到实体的整数 ID |

## 4. 为什么 GraphRAG 还不够

论文和代码都围绕 GraphRAG 的几个局限展开：

### 4.1 社区质量问题

GraphRAG 用 Leiden 之类的图聚类算法划分社区，主要依赖图连接结构。可是知识图谱的边并不总是完美表达语义相似性。

例如两个实体都连到同一个中心实体，但主题可能不同；或者两个主题相近的实体没有直接边。只看图结构，社区容易混杂。

ArchRAG 的改进是：聚类时同时考虑图结构和实体文本属性，也就是实体描述 embedding。

### 4.2 只用一个粒度不够

GraphRAG 的 Global Search 偏高层摘要，适合抽象问题；Local Search 偏实体和文本块，适合具体问题。但真实用户问题往往混合了多个粒度。

ArchRAG 把实体和不同层级社区都放进一个层级索引里，查询时可以同时从多个层级取结果。

### 4.3 Token 成本高

GraphRAG Global Search 可能要遍历很多社区摘要，然后用 LLM 过滤，这在大图上非常贵。

ArchRAG 用 HCHNSW/C-HNSW 做近似最近邻检索，只取最相关的实体和社区，避免全量扫社区。

## 5. ArchRAG 的离线索引阶段

ArchRAG 的离线索引从 GraphRAG 输出开始。

入口是：

```text
src/index.py
```

核心函数是：

```python
make_hc_index(args)
```

### 5.1 读取图

`src/utils.py` 里的 `read_graph_nx()` 读取实体和关系：

```python
graph, entities_df, final_relationships = read_graph_nx(...)
```

它会构建一个 NetworkX 图：

- 节点：实体的 `human_readable_id`
- 边：关系的 `head_id -> tail_id`

### 5.2 实体 embedding

`process_entity_embedding()` 会对实体做 embedding：

```python
entity_content = entity_name + " " + entity_description
```

得到的向量被写入：

```text
entities_df["embedding"]
graph.nodes[node_id]["embedding"]
```

这一步让每个实体不仅有图结构位置，还有语义属性。

### 5.3 属性增强图

在 `src/attr_cluster.py` 中，`compute_distance()` 会基于实体 embedding 增强原图。

它做了两类事：

1. 给已有边计算语义相似度权重。
2. 用 HNSW 找每个节点的近邻，如果两个实体语义相似且超过阈值，就补充新边。

直观理解：

> 原图告诉我们谁和谁有显式关系；embedding 相似度告诉我们谁和谁主题接近。ArchRAG 把两者结合起来做社区。

### 5.4 属性社区聚类

聚类入口是：

```python
attr_cluster(...)
```

默认使用：

```python
compute_leiden_max_size(...)
```

也就是带最大社区大小约束的层次 Leiden。

每一轮聚类会得到若干社区：

```text
community_id -> [entity_id_1, entity_id_2, ...]
```

### 5.5 用 LLM 生成社区报告

每个社区不是只保存一组节点，还会用 LLM 写一份报告：

```text
title
summary
findings
rating
rating_explanation
```

代码在：

```text
src/community_report.py
```

这些社区报告非常重要，因为在线检索时，高层问题主要靠这些摘要提供信息。

### 5.6 社区 embedding

社区报告生成后，项目会对：

```text
title + summary
```

做 embedding，得到社区向量。

于是实体和社区都可以进入同一个向量索引。

### 5.7 重构高层图

完成第 1 层社区后，ArchRAG 会把每个社区看作一个新节点，重新建一张更高层图：

- 节点：社区
- 边：如果两个社区内部实体之间存在关系，则社区之间连边
- 节点属性：社区摘要 embedding

然后继续聚类，形成第 2 层、第 3 层……

这就是“层级社区”的来源。

### 5.8 生成 level summary

`src/client_reasoning.py` 里的 `level_summary()` 会为每一层生成整体说明：

```text
level_summary.csv
```

它用于让系统理解：

- 第 1 层大概是什么粒度
- 第 2 层大概是什么粒度
- 某个问题更应该从哪一层找信息

## 6. HNSW 和 HCHNSW/C-HNSW 是什么

### 6.1 HNSW 的直觉

HNSW 是 Hierarchical Navigable Small World 的缩写，是一种常用的近似最近邻索引。

普通向量检索如果暴力搜索，就是拿 query 向量和所有向量算距离，成本是 `O(N)`。

HNSW 的思路像“城市道路系统”：

- 高层是高速路，节点少，能快速接近目标区域。
- 低层是街道，节点多，能精细找到最近邻。

搜索时先从高层快速接近，再逐层下降，最终在底层找到近邻。

### 6.2 ArchRAG 为什么不用普通 HNSW

普通 HNSW 的层级是随机生成的，每个向量可能出现在多个层。

但 ArchRAG 的层级有真实语义：

- 第 0 层是实体。
- 第 1 层是低层社区。
- 第 2 层是更高层社区。

所以它需要一个“社区层级感知”的 HNSW，也就是论文里的 C-HNSW。代码里叫 HCHNSW。

### 6.3 本仓库里的 HCHNSW

自定义 Faiss 代码在：

```text
HCHNSW/faiss/
```

Python 侧调用在：

```text
src/hchnsw_index.py
```

关键代码：

```python
index = faiss.IndexHCHNSWFlat(dim, ML, M, 1, vector_size)
index.set_vector_level(levels)
index.add(embeddings)
```

其中：

- `dim`：embedding 维度。
- `ML`：最大层级。
- `M`：每个节点连接的近邻数。
- `levels`：每个向量属于哪一层。

### 6.4 实体和社区如何放入索引

`get_vector_hchnsw()` 会把两类向量拼起来：

```text
community_embeddings + entity_embeddings
```

并设置层级：

```text
entity level = 0
community level = community_df["level"]
```

然后给每个实体和社区分配 `index_id`。在线检索返回的就是这些 `index_id`。

## 7. 在线检索阶段

入口在：

```text
src/inference.py
```

主要函数：

```python
hcarag(query_content, index_dict, query_paras, args)
```

### 7.1 加载索引

`load_index()` 会读取：

```text
hchnsw.index
entity_df_index.csv
community_df_index.csv
relationship_df_index.csv
relationship_embedding.csv
level_summary.csv
```

并重新读取 GraphRAG 的实体/关系图，用于可选 PPR 扩展。

### 7.2 查询 embedding

系统先把用户问题变成向量：

```python
query_embedding = openai_embedding(query_content, ...)
```

### 7.3 决定每层取多少结果

`load_strategy()` 支持几种策略：

| 策略 | 含义 |
| --- | --- |
| `global` | 每层固定取 `k_each_level` 个 |
| `all` | 类似 GraphRAG，取指定范围层级 |
| `inference` | 先让 LLM 判断问题更适合哪些层，再分配 top-k |

当前扩展 demo 使用：

```text
strategy = global
k_each_level = 6
k_final = 12
```

### 7.4 分层搜索 HCHNSW

代码会对每一层设置搜索参数：

```python
params = faiss.SearchParametersHCHNSW()
params.search_level = level
hc_index.search(query_embedding, k=..., params=params)
```

这样它可以分别取：

- 实体层结果
- 第 1 层社区结果
- 第 2 层社区结果

### 7.5 分离实体和社区

返回的 `index_id` 会分别在两个表中查：

```python
topk_entity = entity_df[entity_df["index_id"].isin(final_predictions)]
topk_community = community_df[community_df["index_id"].isin(final_predictions)]
```

### 7.6 关系补充

如果只拿实体和社区，可能缺少关系细节。所以系统会：

1. 找出命中实体发出的关系。
2. 对关系描述 embedding 和 query embedding 算相似度。
3. 取最相关的 `topk_e` 条关系。

输出是：

```text
topk_related_r
```

### 7.7 可选 PPR

如果打开：

```text
ppr_refine = true
```

系统会以初始命中实体为 personalization，用 PageRank 在图上扩展相关实体。

当前 demo 为了简单和稳定，没有打开 PPR。

## 8. 答案生成

项目支持两种生成方式。

### 8.1 direct

把检索到的实体、关系、社区摘要拼成一个 prompt，直接让 LLM 回答。

优点：简单、快。  
缺点：上下文多时容易丢重点。

### 8.2 mr

当前 demo 使用 `mr`，也就是 map-reduce。

流程：

1. 先让 LLM 自己根据问题做一个初始推理。
2. 把实体/关系上下文、社区上下文、可选文本块上下文分成多个 chunk。
3. map 阶段：每个 chunk 让 LLM 提取 key points，并打重要性分数。
4. reduce 阶段：把 key points 排序、合并，再生成最终答案。

这个方法更贵，但对长上下文更稳。

## 9. 本轮扩展 demo 的运行结果

当前语料：

```text
corpus/input/
```

共有 12 篇文档，主题是 Aurora Harbor Resilience Program。

QA 文件：

```text
corpus/expanded_demo_qa.jsonl
```

GraphRAG 输出：

```text
corpus/output/20260520-214913/artifacts/
```

本轮 GraphRAG 结果：

```text
text units: 12
entities: 62
relationships: 139
```

ArchRAG 输出目录：

```text
expanded_archrag_real_index/
```

关键文件：

```text
expanded_archrag_real_index/hchnsw.index
expanded_archrag_real_index/entity_df_index.csv
expanded_archrag_real_index/community_df_index.csv
expanded_archrag_real_index/relationship_df_index.csv
expanded_archrag_real_index/relationship_embedding.csv
expanded_archrag_real_index/level_summary.csv
```

运行脚本：

```text
scripts/run_expanded_archrag_real.py
```

## 10. 复跑命令

从仓库根目录运行：

```bash
cd /home/qrq/projects/ArchRAG
```

重新跑 GraphRAG：

```bash
PYTHONPATH=$(pwd)/src:$(pwd) conda run -n archrag \
  python -m graphrag.index --root ./corpus --emit parquet,csv --reporter print
```

重新跑 ArchRAG：

```bash
PYTHONPATH=$(pwd)/src:$(pwd) conda run -n archrag \
  python scripts/run_expanded_archrag_real.py
```

如果只想指定某个 GraphRAG artifacts：

```bash
PYTHONPATH=$(pwd)/src:$(pwd) conda run -n archrag \
  python scripts/run_expanded_archrag_real.py \
  --artifacts corpus/output/20260520-214913/artifacts
```

## 11. 如何读懂几个输出表

### 11.1 `entity_df_index.csv`

这是实体表。重点看：

| 字段 | 说明 |
| --- | --- |
| `name` | 实体名 |
| `description` | 实体描述 |
| `embedding` | ArchRAG 重新生成的实体向量 |
| `index_id` | 进入 HCHNSW 后的统一索引 ID |

### 11.2 `community_df_index.csv`

这是社区表。重点看：

| 字段 | 说明 |
| --- | --- |
| `title` | LLM 生成的社区标题 |
| `summary` | 社区摘要 |
| `community_nodes` | 社区包含的底层实体 ID |
| `level` | 社区层级 |
| `embedding` | 社区摘要向量 |
| `index_id` | 进入 HCHNSW 后的统一索引 ID |

### 11.3 `relationship_df_index.csv`

这是关系表。重点看：

| 字段 | 说明 |
| --- | --- |
| `source` / `target` | 关系两端实体 |
| `description` | 关系文字描述 |
| `source_index_id` / `target_index_id` | 关系两端实体在 HCHNSW 中的 ID |
| `embedding_idx` | 关系 embedding 的引用 ID |

### 11.4 `relationship_embedding.csv`

关系描述的 embedding 表。为了避免重复存储，每条关系通过 `embedding_idx` 映射到这里。

### 11.5 `level_summary.csv`

每个层级的总体说明。它可以帮助系统理解不同层级的语义粒度。

## 12. 项目中的关键参数

### 12.1 离线索引参数

| 参数 | 含义 |
| --- | --- |
| `max_level` | 最多构建多少层社区 |
| `min_clusters` | 如果某层聚类数量低于该值，就停止继续向上构建 |
| `max_cluster_size` | 单个低层社区最大节点数 |
| `wx_weight` | 属性相似度阈值的分位数 |
| `search_k` | 查找相似邻居时的扩展比例 |
| `m_du_scale` | 允许补边数量的缩放系数 |
| `augment_graph` | 是否用 embedding 相似度增强图 |
| `cluster_method` | 聚类方法，当前主要用 `weighted_leiden` |

### 12.2 在线检索参数

| 参数 | 含义 |
| --- | --- |
| `strategy` | 层级检索策略 |
| `k_each_level` | 每层取多少候选 |
| `k_final` | 最终保留多少结果 |
| `topk_e` | 取多少相关关系 |
| `generate_strategy` | `direct` 或 `mr` |
| `ppr_refine` | 是否用 PageRank 扩展实体 |
| `topk_chunk` | 是否额外使用普通文本 chunk 检索 |

## 13. 从代码角度看完整链路

完整链路可以理解为：

```text
corpus/input/*.txt
  -> python -m graphrag.index
  -> create_final_entities.parquet
  -> create_final_relationships.parquet
  -> src/index.py
  -> read_graph_nx()
  -> entity_embedding()
  -> attr_cluster()
  -> community_report_batch()
  -> create_hchnsw_index()
  -> hchnsw.index + csv tables
  -> src/inference.py
  -> load_index()
  -> hcarag_retrieval()
  -> hcarag_inference()
  -> final answer
```

## 14. 初学者理解重点

如果你刚接触这篇论文和项目，可以先抓住五件事：

1. GraphRAG 负责从文本中抽知识图谱。
2. ArchRAG 不满足于 GraphRAG 的原始社区，而是用实体属性 embedding 增强图聚类。
3. 社区不是简单节点集合，而是 LLM 总结后的语义单元。
4. HCHNSW 把实体和多层社区放进同一个层级向量索引。
5. 在线回答时，系统从多个层级取实体、关系、社区摘要，再用 LLM map-reduce 生成答案。

这就是项目名字里几个词的含义：

```text
Attributed     -> 使用实体/社区文本属性和 embedding
Community      -> 用社区作为中高层知识单元
Hierarchical   -> 社区逐层聚合，形成层级
RAG            -> 检索增强生成
```
