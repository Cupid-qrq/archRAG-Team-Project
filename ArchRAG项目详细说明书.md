# ArchRAG 项目详细说明书

## 一、项目概述

ArchRAG（**A**ttributed **C**ommunity-based Hie**r**archical **R**etrieval-**A**ugmented **G**eneration）是一种基于属性社区层次化组织的图增强 RAG 系统。项目对应论文"[ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation](https://arxiv.org/abs/2502.09891)"的完整开源实现。

### 核心目标

解决传统 RAG 和微软 GraphRAG 的三个核心局限：

| GraphRAG 局限 | ArchRAG 应对策略 | 效果 |
|--------------|-----------------|------|
| L1：社区质量低（Leiden 只看图结构） | 属性社区检测（结构 + 语义双重约束） | 社区内语义一致性显著提升 |
| L2：单粒度检索，无法适配不同抽象层次的问题 | 迭代层次聚类 + C-HNSW 跨层索引 | 同时支持具体事实和抽象概括问题 |
| L3：Global Search 遍历所有社区，Token 成本高 | 分层检索 + LLM 自适应过滤 | Token 消耗降至 1/250，速度提升 5.4x |

### 最终效果

- **准确率**：比 GraphRAG SOTA 高 10%+
- **Token 成本**：仅为 GraphRAG Global Search 的 1/250
- **检索速度**：C-HNSW 比"每层独立 HNSW"快 3.5-5.4 倍
- **社区质量**：CHI 指数和余弦相似度均显著优于 Leiden

---

## 二、四大创新点详解

### 创新点一：属性社区（Attributed Communities）检测

**解决的问题**：GraphRAG 的 Leiden 社区检测仅基于图拓扑结构，忽略节点自身的文本语义，导致检测出的社区可能包含不同主题的节点，社区摘要质量差。

**实现方案**：

1. **图增强（Graph Augmentation）**：计算节点属性向量（实体描述 embedding）之间的余弦相似度，用 HNSW 搜索 K 近邻，对相似度超过分位数阈值的节点对添加"虚拟边"
2. **加权聚类**：每条边（含原始边和增强边）权重视为属性相似度，运行加权 Leiden 算法进行社区检测
3. **LLM 生成摘要**：对每个属性社区，将其内部所有实体的描述和关系作为上下文，调用 LLM 生成概括性文本

**源码位置**：`src/attr_cluster.py` — `compute_distance()` / `compute_leiden_max_size()`  
**源代码入口**：`src/community_report.py` — `community_report_batch()`

### 创新点二：基于 LLM 的迭代层次聚类

**解决的问题**：单层社区只能提供单一粒度信息，具体问题需要细粒度实体，抽象问题需要高层主题概括。

**实现方案**（算法流程）：

1. **初始层 L0**：原始 KG，节点 = 实体，边 = 关系
2. 对当前图进行图增强（KNN 加边）
3. 计算边权重 = 节点属性相似度
4. 运行加权图聚类 → 得到一组属性社区
5. **LLM 生成摘要**（成为该社区的文本属性）
6. **构建高一层的图**：每个社区收缩为超节点，超节点间根据原图跨社区边连边
7. 重复直到满足停止条件（节点数 < `min_clusters` 或达到 `max_level`）

最终得到一棵层次语义树 Δ：根节点是顶层全局社区，叶节点是原始实体，每个节点有 LLM 生成的文本摘要。

**源码位置**：`src/attr_cluster.py` — `attr_cluster()` 主循环 + `reconstruct_graph()` 高层图构建  
**停止条件**：`max_level`（默认 4）、`min_clusters`（默认 5）

### 创新点三：C-HNSW 跨层统一向量索引

**解决的问题**：层次社区树有多个层级，朴素方法为每层独立建索引会导致存储/查询开销大，且无法跨层跳跃检索。

**实现方案**：

1. **节点集合**：所有层的社区节点 + 实体节点统一纳入一个多层图结构
2. **分层组织**：Layer 0 = 实体，Layer 1 = 第一层社区，Layer 2 = 第二层社区...（与社区树的语义层级严格对齐）
3. **跨层链接（Inter-layer links）**：若低层实体属于某高层社区，则在两者间建立双向跨层边
4. **查询过程**：从最高层入口点开始 → 同层贪心搜索 → 跨层链接下降到下一层 → 重复至 L0 → 返回 Top-K

**源码位置**：`HCHNSW/faiss/`（C++ 实现，基于 Faiss v1.8.0 修改）  
**Python 封装**：`src/hchnsw_index.py` — `create_hchnsw_index()` / `faiss.IndexHCHNSWFlat`

**与普通 HNSW 的核心区别**：

| 维度 | 普通 HNSW | C-HNSW |
|------|-----------|--------|
| 层级结构 | 随机子采样，无语义 | 与社区树语义层级严格对应 |
| 跨层链接 | 仅连接同一节点在不同层的副本 | 连接不同节点（社区与成员），反映包含关系 |
| 查询语义 | 纯向量空间最近邻 | 从粗到细的语义导航 |

### 创新点四：自适应过滤的分层检索

**解决的问题**：检索结果可能包含噪声、冗余、粒度不匹配的信息，直接拼接生成会浪费 Token 且干扰 LLM。

**实现方案**（采用 Map-Reduce 策略）：

1. **Map 阶段（重要性评分）**：将检索到的实体、社区、关系分成多个 chunk，每个 chunk 调用 LLM 提取 key points 并打分（0-100）
2. **过滤**：只保留分数 > 0 的 points，按分数降序排列，控制总 Token 不超出限制
3. **Reduce 阶段（融合生成）**：将过滤后的高价值 points 拼接为上下文，连同用户问题输入 LLM 生成最终答案

**源码位置**：`src/client_reasoning.py` — `map_llm_worker()`（Map 评分）、`reduce_inference()`（Reduce 融合）  
**调用入口**：`src/inference.py` — `hcarag_inference_mr()`

---

## 三、项目目录结构

```
archRAG-Team-Project/
├── HCHNSW/                          # 修改版 Faiss（C-HNSW 的 C++ 实现）
│   ├── faiss/                       # Faiss v1.8.0 源码，新增 IndexHCHNSW
│   └── test_mycode/                 # C-HNSW 测试代码
│
├── src/                             # 核心 Python 代码
│   ├── index.py                     # 离线索引构建入口（make_hc_index）
│   ├── inference.py                 # 在线检索/推理入口（hcarag）
│   ├── attr_cluster.py              # 属性社区检测 + 迭代层次聚类
│   ├── hchnsw_index.py              # C-HNSW 索引构建（调用 faiss.IndexHCHNSWFlat）
│   ├── community_report.py          # LLM 社区摘要生成（批量并行）
│   ├── client_reasoning.py          # 自适应过滤 + 层级摘要 + Map-Reduce 推理
│   ├── llm.py                       # LLM 调用封装（OpenAI 兼容 API）
│   ├── lm_emb.py                    # Embedding 模型封装（远程API + 本地 SBERT）
│   ├── prompts.py                   # 所有 Prompt 模板（6大类）
│   ├── utils.py                     # 工具函数（图读写、embedding、图增强等）
│   ├── ppr_entity_search.py         # PPR（Personalized PageRank）实体扩展搜索
│   ├── __init__.py                  # 包初始化
│   └── evaluate/                    # 评估模块
│       ├── evaluate.py              # 主评估脚本（Accuracy/F1/BLEU/METEOR/ROUGE）
│       ├── test_qa.py               # QA 测试
│       ├── community_evaluate.py    # 社区质量评估（CHI、余弦相似度）
│       ├── attr_cluster_metric.py   # 属性聚类质量指标
│       ├── leiden_origin_graph_community_evaluate.py  # Leiden vs AC 对比
│       ├── baseline_node2vec.py     # Node2Vec 基线
│       ├── query_generation.py      # 查询生成
│       ├── query_generation_v2.py   # 查询生成 v2
│       ├── description_generation.py # 描述生成
│       ├── hchnsw_evaluate.py       # C-HNSW 性能评估（速度/召回率/内存）
│       └── summary_eval.py          # 摘要质量评估
│
├── src/graphrag/                    # 微软 GraphRAG 修改版（KG 提取用）
│   ├── config/                      # GraphRAG 配置
│   ├── llm/                         # LLM 调用
│   ├── model/                       # 数据模型（实体/关系/社区/文档等）
│   └── prompt_tune/                 # Prompt 调优
│
├── corpus/                          # 示例语料
│   └── input/                       # 原始文本文档（Aurora Harbor 城市韧性项目）
│
├── dataset/                         # 数据集脚本
│   ├── index.sh                     # 离线索引构建脚本
│   ├── query.sh                     # 在线检索脚本
│   └── settings.yaml                # GraphRAG 配置文件
│
├── scripts/                         # 运行脚本
│   ├── run_tiny_archrag_demo.py     # 小规模 Demo
│   ├── run_tiny_archrag_real.py     # 真实数据运行
│   ├── run_expanded_archrag_real.py # 扩展语料运行脚本
│   └── query_expanded_archrag_real.py # 扩展查询脚本
│
├── docs/                            # 项目文档
│   ├── ArchRAG_论文与仓库分析_创新局限改进.html
│   ├── ArchRAG_论文与仓库分析_创新局限改进.md
│   └── ArchRAG_项目原理与全流程指南.md
│
├── json/                            # JSON 数据目录
├── metric/                          # 评估指标目录
├── requirements.txt                 # Python 依赖
├── .gitignore                       # Git 忽略规则
└── README.md                        # 项目 README
```

---

## 四、数据流全链路

### 阶段一：知识图谱提取（Microsoft GraphRAG）

```
原始文本 (*.txt)
  → python -m graphrag.index
  → 文本切块 → 实体抽取 → 关系抽取 → 实体/关系合并
  → 输出:
      create_final_entities.parquet       (实体表)
      create_final_relationships.parquet  (关系表)
```

### 阶段二：ArchRAG 离线索引构建

```
入口: src/index.py → make_hc_index(args)

Step 1. read_graph_nx()
        读取 entities.parquet + relationships.parquet
        → 构建 NetworkX 图 (节点=human_readable_id, 边=head_id→tail_id)

Step 2. process_entity_embedding()
        对每个实体 "name + description" 做 embedding
        → entities_df["embedding"]
        → graph.nodes[id]["embedding"]

Step 3. attr_cluster() 迭代循环 (level=1 to max_level):
        3a. compute_distance()
            已有边 → 计算语义相似度作为权重
            HNSW 搜索 K 近邻 → 相似度 > wx 分位数阈值 → 加虚拟边
        3b. compute_leiden_max_size()
            加权 Leiden 聚类 (带 max_cluster_size 约束)
            → community_id → [entity_ids]
        3c. community_report_batch()
            并行调用 LLM 为每个社区生成摘要
            → {title, summary, findings, rating, rating_explanation}
        3d. report_embedding()
            对社区 "title + summary" 做 embedding
        3e. reconstruct_graph()
            每个社区收缩为超节点
            超节点间根据原图跨社区边连边
            → 得到高一层的图，作为下一轮迭代的输入
        终止条件: len(communities) < min_clusters OR level >= max_level

Step 4. create_hchnsw_index()
        收集所有层节点的向量 (实体向量 + 各层社区向量)
        设置层级: entity level=0, community level=实际层级
        → faiss.IndexHCHNSWFlat(dim, max_level, M=32, ...)
        → index.set_vector_level(levels)
        → index.add(embeddings)
        → 保存 hchnsw.index + 各 CSV 表

Step 5. make_level_summary()
        对每层抽样社区 → LLM 生成该层整体说明
        → level_summary.csv
```

**输出文件**：

| 文件 | 内容 |
|------|------|
| `hchnsw.index` | C-HNSW 向量索引 |
| `entity_df_index.csv` | 实体表（含 name/description/embedding/index_id） |
| `community_df_index.csv` | 社区表（含 title/summary/level/community_nodes/embedding/index_id） |
| `relationship_df_index.csv` | 关系表（含 source/target/description/source_index_id/target_index_id） |
| `relationship_embedding.csv` | 关系描述 embedding |
| `level_summary.csv` | 每层整体摘要 |

### 阶段三：ArchRAG 在线检索与生成

```
入口: src/inference.py → hcarag(query, index_dict, query_paras, args)

Step 1. load_index()
        加载 hchnsw.index + 所有 CSV 表
        → index_dict

Step 2. hcarag_retrieval()
        2a. 查询向量化: openai_embedding(query)
        2b. 决定检索策略:
            - global: 每层固定取 k_each_level 个
            - inference: LLM 先判断问题适合哪层，再分配 k 值
            - all: 类似 GraphRAG，取指定范围层级
        2c. 分层 C-HNSW 搜索:
            for level in range(max_level):
                params.search_level = level
                hc_index.search(query_embedding, k=k_per_level[level], params)
            → 合并各层结果 → 按距离排序 → 取 Top-K
        2d. 分离结果:
            按 index_id 在 entity_df/community_df 中匹配
            → topk_entity, topk_community
        2e. 关系补充:
            找出命中实体相关的所有关系
            关系描述 embedding 与 query 算余弦相似度
            → topk_related_r (Top-K 相关关系)
        2f. 可选 PPR 扩展 (ppr_refine):
            以命中实体为 personalization
            在原始 KG 上运行 PageRank
            → 扩展获取更多相关实体
        2g. 可选 Chunk 检索 (topk_chunk):
            在传统文本块向量索引中额外检索

Step 3. hcarag_inference()
        direct 模式:
            拼接检索结果 → 直接 LLM 生成答案
        mr 模式 (Map-Reduce):
            Map: 将实体/社区/关系分成多个 chunk
                 每个 chunk → LLM 提取 key points + 打分(0-100)
            Filter: 过滤 score≤0 的 points，按 score 降序排列
            Reduce: 拼接高价值 points → LLM 融合生成最终答案
```

---

## 五、关键模块详解

### 5.1 图增强（`compute_distance`）

**位置**：`src/utils.py`

**流程**：
1. 遍历原图所有边，用两端节点的 embedding 计算余弦相似度作为边权重
2. 计算所有权重的 x-分位数作为阈值 `wx`
3. 计算平均度数 `m_du`，决定每节点最多添加多少条新边
4. 对每个节点，用 HNSW 搜索 `search_k * m_du` 个近邻
5. 若近邻的余弦相似度 > wx 且无边，添加新边（权重=相似度）
6. 每节点最多保留 `m_du` 条最优新边

**参数**：`wx_weight`（分位数阈值，默认 0.7）、`search_k`（搜索放大系数，默认 1.5）、`m_du_scale`（度数缩放，默认 1）

### 5.2 聚类方法

**位置**：`src/attr_cluster.py`

支持两种聚类算法：

- **加权 Leiden**（默认）：`compute_leiden_max_size()` — 使用 `graspologic.partition.hierarchical_leiden`，带 `max_cluster_size` 约束
- **GPU 谱聚类**：`spectral_clustering_cupy()` — 使用 CuPy 计算拉普拉斯矩阵、特征分解，再用 KMeans 聚类，适用于大规模图

### 5.3 C-HNSW 索引

**C++ 层**（`HCHNSW/faiss/`）：
- 在 Faiss v1.8.0 基础上新增 `IndexHCHNSWFlat` 类
- 魔改 HNSW 的层级分配逻辑：层由社区树决定（非随机）
- 增加跨层链接：低层节点指向包含它的高层社区节点

**Python 封装**（`src/hchnsw_index.py`）：
```python
index = faiss.IndexHCHNSWFlat(dim, ML, M, 1, vector_size)
index.set_vector_level(levels)  # 注入语义层级
index.hchnsw.efSearch = 40
index.hchnsw.efConstruction = 16
index.add(embeddings)
```

**参数**：`M=32`（每节点连接数）、`efSearch=40`（查询时搜索宽度）、`efConstruction=16`（构建时搜索宽度）

### 5.4 LLM 调用封装

**位置**：`src/llm.py`

- 使用 OpenAI 兼容 API（`client.chat.completions.create`）
- 支持 JSON 模式输出（`response_format: json_object`）
- 自动重试机制（`max_retries`）
- 支持 Ollama 本地模型（`api_key="ollama"`）和 vLLM 等兼容服务

### 5.5 Embedding 模块

**位置**：`src/lm_emb.py`

支持两种模式：
- **远程 API**：`openai_embedding()` — 调用 OpenAI 兼容 API
- **本地模型**：`text_to_embedding_batch()` — 加载 Sentence-Transformer（默认 `nomic-embed-text-v1`），支持 DataParallel 多 GPU

### 5.6 Prompt 体系

**位置**：`src/prompts.py`

定义了完整的 Prompt 体系：

| Prompt 常量 | 用途 | 调用位置 |
|------------|------|---------|
| `COMMUNITY_REPORT_PROMPT` | 生成社区摘要（title/summary/findings/rating） | `community_report.py` |
| `LEVEL_SUMMARY_PROMPT` | 生成每层整体摘要 | `client_reasoning.py` |
| `LEVEL_INFERENCE_PROMPT` | LLM 判断问题适合哪一层 | `client_reasoning.py` |
| `GLOBAL_MAP_SYSTEM_PROMPT` | Map 阶段：提取 key points 并打分 | `client_reasoning.py` |
| `GLOBAL_REDUCE_SYSTEM_PROMPT` | Reduce 阶段：融合高价值信息生成答案 | `client_reasoning.py` |
| `GENERATION_PROMPT` | Direct 模式：一步生成答案 | `client_reasoning.py` |

---

## 六、参数配置

### 6.1 离线索引参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_level` | int | 4 | 最大社区层级数 |
| `min_clusters` | int | 5 | 停止条件：顶层社区数低于此值 |
| `max_cluster_size` | int | 15 | 单个社区最大节点数 |
| `wx_weight` | float | 0.7 | 图增强：相似度阈值分位数 |
| `search_k` | float | 1.5 | 图增强：HNSW 搜索放大系数 |
| `m_du_scale` | float | 1 | 图增强：允许补边数量缩放 |
| `augment_graph` | bool | true | 是否启用属性图增强 |
| `cluster_method` | str | "weighted_leiden" | 聚类方法（weighted_leiden / spectral） |
| `seed` | int | 0xDEADBEEF | 随机种子（可复现性） |

### 6.2 在线检索参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | str | "global" | 检索策略：global / all / inference |
| `k_each_level` | int | 5 | 每层检索候选数 |
| `k_final` | int | 15 | 最终保留结果数 |
| `topk_e` | int | 10 | 相关关系取 Top-K |
| `generate_strategy` | str | "mr" | 生成策略：direct / mr (Map-Reduce) |
| `ppr_refine` | bool | false | 是否启用 PPR PageRank 扩展 |
| `topk_chunk` | int | 0 | 额外文本块检索数（0=不启用） |
| `only_entity` | bool | false | 仅检索实体层 |
| `wo_hierarchical` | bool | false | 不使用层次社区 |

大模型及 Embedding 相关参数统一见 [第九节](#九大模型与-embedding-参数配置)。

---

## 七、评估体系

**位置**：`src/evaluate/`

### 评估数据集

支持 8+ 个数据集：HotpotQA、MultiHopRAG、NarrativeQA、WebQSP、Mintaka、PopQA、WebQ、RAG-QA-Arena 系列

### 评估指标

- **KGQA 模式**（知识图谱问答）：Accuracy、Hit、F1、Precision、Recall
- **DocQA 模式**（文档问答）：Hit、F1、Precision、Recall、Exact Match (EM)
- **生成质量**：BLEU-1/4、METEOR、ROUGE-L
- **社区质量**：CHI 指数、社区内余弦相似度
- **检索性能**：C-HNSW 速度 / 召回率 / 内存对比

### 消融实验设计

| 变体 | 说明 | 对应代码控制 |
|------|------|-------------|
| Direct Prompt | 去掉自适应过滤，直接拼接检索结果 | `generate_strategy="direct"` |
| Single-Layer | 去掉层次聚类，只保留单层 | `max_level=1` |
| No AC | 去掉属性增强，仅用图结构聚类 | `augment_graph=false` |

---

## 八、快速上手

### 环境准备

```bash
# 创建环境
conda create -n archrag python=3.10 -y
conda activate archrag

# 安装依赖
pip install -r requirements.txt

# 编译自定义 Faiss (C-HNSW)
cd HCHNSW
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)

# 设置 Python 路径
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 离线索引

```bash
# Step 1: GraphRAG 提取 KG
PYTHONPATH=$(pwd)/src:$(pwd) python -m graphrag.index \
    --root ./corpus --emit parquet,csv --reporter print

# Step 2: ArchRAG 构建索引
bash dataset/index.sh
# 或
python src/index.py --base_path <artifacts_path> --output_dir <output_path>
```

### 在线检索

```bash
bash dataset/query.sh
# 或
python src/inference.py --output_dir <index_path> --dataset_name hotpot
```

---

## 九、大模型与 Embedding 参数配置

本项目默认配置为本地部署（Ollama + 本地 Embedding）。本节统一说明所有大模型和 Embedding 相关的参数、默认值，以及迁移到云端 API 的完整步骤。

### 参数一览

#### LLM 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `engine` | str | `"llama3.1:8b4k"` | LLM 模型名称 |
| `api_key` | str | `"ollama"` | API Key |
| `api_base` | str | `"http://localhost:5001/forward"` | API Base URL |
| `max_tokens` | int | 4000 | 单次生成最大 Token 数 |
| `temperature` | float | 0.1 | LLM 温度（控制输出随机性） |
| `max_retries` | int | 5 | API 调用失败最大重试次数 |
| `max_community_tokens` | int | 4000 | 社区报告生成最大 Token 数 |

#### Embedding 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_model` | str | `"nomic-embed-text"` | 远程 Embedding 模型名称 |
| `embedding_api_key` | str | `"ollama"` | Embedding API Key |
| `embedding_api_base` | str | `"http://localhost:5001/forward"` | Embedding API Base URL |
| `embedding_local` | bool | false | 是否使用本地 Embedding 模型 |
| `embedding_model_local` | str | `"nomic-embed-text-v1"` | 本地 Embedding 模型名称 |
| `embedding_num_workers` | int | 32 | Embedding 并行线程/进程数 |
| `entity_second_embedding` | bool | true | 是否对实体做二次 Embedding |

> **注意**：`src/utils.py` 中 argparse 定义的以上默认值（如 `api_key="ollama"`, `api_base="http://localhost:5001/forward"` 等）无需修改，shell 脚本传入的参数会自动覆盖这些默认值。

### 迁移到云端 API

若要将本地 Ollama 替换为云端 API，需修改以下 **4 个文件 + 1 行代码**。

项目有两套独立的配置体系，分别对应不同阶段：

```
GraphRAG 阶段 (KG 提取)          ArchRAG 阶段 (索引构建 + 检索推理)
─────────────────────────        ──────────────────────────────────
读取: settings.yaml              读取: shell 脚本变量 → argparse 命令行参数
      ↑                                ↑
      corpus/settings.yaml              dataset/index.sh, query.sh
      dataset/settings.yaml             ↓
                                   传给 src/llm.py, src/lm_emb.py
```

### 需修改的文件清单

| # | 文件 | 修改内容 |
|---|------|---------|
| 1 | `corpus/settings.yaml` | LLM + Embedding 的 api_key / api_base / model |
| 2 | `dataset/settings.yaml` | 同上 |
| 3 | `dataset/index.sh` | 第 11-20 行的 api_key、api_base、engine、embedding 变量 |
| 4 | `dataset/query.sh` | 第 5-7 行的 api_key、api_base、engine 变量 |
| 5 | `src/llm.py` 第 16-17 行 | 删除或注释 `if "llama" in engine: api_key = "ollama"` |

### 详细修改步骤

#### 1. `corpus/settings.yaml` — GraphRAG LLM 配置

```yaml
llm:
  api_key: <你的LLM_API_KEY>
  type: openai_chat
  model: <你的LLM模型名，如 gpt-4o, deepseek-chat>
  api_base: <你的LLM_API地址>

embeddings:
  llm:
    api_key: <你的Embedding_API_KEY>
    type: openai_embedding
    model: <你的Embedding模型名，如 text-embedding-3-small>
    api_base: <你的Embedding_API地址>
```

#### 2. `dataset/settings.yaml` — 同上，占位符替换为实际值

#### 3. `dataset/index.sh` — ArchRAG 离线索引 API 变量

```bash
api_key="<你的LLM_API_KEY>"
api_base="<你的LLM_API地址>"
engine="<你的LLM模型名>"

embedding_model="<你的Embedding模型名>"
embedding_api_key="<你的Embedding_API_KEY>"
embedding_api_base="<你的Embedding_API地址>"
```

#### 4. `dataset/query.sh` — ArchRAG 在线检索 API 变量

```bash
api_key="<你的LLM_API_KEY>"
api_base="<你的LLM_API地址>"
engine="<你的LLM模型名>"
```

#### 5. `src/llm.py` 第 16-17 行 — 移除 Ollama 强制覆盖

修改前：
```python
if "llama" in args.engine.lower():
    api_key = "ollama"
    base_url = args.api_base
else:
    api_key = args.api_key
    base_url = args.api_base
```

修改后（删除 if-else，统一使用传入参数）：
```python
api_key = args.api_key
base_url = args.api_base
```

> 原逻辑：当模型名包含 "llama" 时，强制将 api_key 替换为 "ollama"，导致云端 API Key 被覆盖，请求认证失败。

### 注意事项

- `src/evaluate/` 目录下的评估脚本（`query_generation_v2.py`、`description_generation.py` 等）含独立的硬编码 API 配置，使用评估功能时需单独修改。
- 以上配置默认使用 OpenAI 兼容 API 格式，支持 OpenAI、DeepSeek、硅基流动、OpenRouter、vLLM 等所有兼容服务。

---

## 十、各模块依赖关系图

```
                    ┌─────────────────────────────────┐
                    │         src/prompts.py           │
                    │   (所有 LLM Prompt 模板定义)      │
                    └──────────┬──────────────────────┘
                               │ 被引用
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼
    ┌───────────────┐  ┌──────────────┐  ┌────────────────────┐
    │ community_    │  │ client_      │  │ prompts 中的        │
    │ report.py     │  │ reasoning.py │  │ GENERATION_PROMPT   │
    │ (社区报告生成) │  │ (自适应过滤)   │  │ (Direct 模式)       │
    └───┬───────────┘  └──────┬───────┘  └─────────┬──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │ 调用
                              ▼
                    ┌──────────────────┐
                    │   src/llm.py     │
                    │ (OpenAI API 封装) │
                    └──────────────────┘

┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  utils.py    │────▶│ attr_cluster.py │────▶│ hchnsw_index.py  │
│ (图读写/增强) │     │ (层次聚类主循环) │     │ (C-HNSW 索引构建) │
└──────┬───────┘     └────────┬────────┘     └────────┬─────────┘
       │                      │                       │
       │                      ▼                       │
       │              ┌───────────────┐               │
       │              │ community_    │               │
       │              │ report.py     │               │
       │              │ (LLM摘要生成)  │               │
       │              └───────────────┘               │
       │                      │                       │
       └──────────────────────┼───────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   index.py       │
                    │ (离线索引入口)    │
                    └──────────────────┘
                              │
                              │ 生成
                              ▼
              ┌───────────────────────────┐
              │  hchnsw.index + CSV 表     │
              └───────────────────────────┘
                              │
                              │ 加载
                              ▼
                    ┌──────────────────┐
                    │  inference.py    │
                    │ (在线检索入口)    │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            ┌──────────────┐  ┌──────────────────┐
            │ hcarag_      │  │ hcarag_inference  │
            │ retrieval()  │  │ (direct / mr)     │
            │ (C-HNSW检索) │  │                  │
            └──────────────┘  └──────────────────┘
```

---

## 十一、总结

ArchRAG 是一个完整的、自成体系的图增强 RAG 系统。其核心设计思路是：**用属性语义增强社区检测，用层次摘要树提供多粒度信息，用 C-HNSW 实现高效跨层检索，用 LLM 自适应过滤提升答案精度**。四大创新点相互支撑、缺一不可，共同构成了"离线索引 + 在线检索"的完整解决方案。

- **属性社区**：保证信息块内部语义一致性
- **层次聚类**：提供多抽象层次的语义树
- **C-HNSW**：将语义层次编码进向量索引，实现单次查询跨层导航
- **自适应过滤**：动态去除噪声，Token 高效且答案精准
