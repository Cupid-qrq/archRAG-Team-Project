"""HyperNode: explicit multi-hop path representation for graph retrieval."""

from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class HyperNode:
    triple_ids: list
    path_embedding: np.ndarray
    similarity: float

    def serialize(self) -> str:
        return f"hyper_{'_'.join(map(str, sorted(self.triple_ids)))}"


def generate_hypernodes(
    query_embedding: np.ndarray,
    relation_df: pd.DataFrame,
    graph: nx.Graph,
    topk_seeds: int = 10,
    max_hops: int = 2,
    topk_per_hop: int = 10,
) -> tuple[list[HyperNode], dict]:
    """Generate multi-hop paths by expanding from seed triples along the KG.

    Args:
        query_embedding: (1, dim) or (dim,) query vector
        relation_df: must have columns [head_id, tail_id, embedding, source_index_id, target_index_id]
        graph: NetworkX graph (nodes keyed by human_readable_id)
        topk_seeds: number of seed triples by embedding similarity
        max_hops: max path length (1 = single triple, 2 = triple→triple, etc.)
        topk_per_hop: max paths retained after each hop expansion

    Returns:
        (hypernodes, triple_map):
        - hypernodes: List of HyperNode, sorted by similarity desc.
        - triple_map: dict mapping _tid → row Series, for downstream score computation.
    """
    if relation_df.empty:
        return [], {}

    # 加一列 _tid 作为函数内部的 triple 唯一标识，不依赖 DataFrame index
    # 同时建一个 tid→row 的 dict，供后续 helper 函数 O(1) 查询
    relation_df = relation_df.copy()
    relation_df["_tid"] = range(len(relation_df))
    _triple_map = {row["_tid"]: row for _, row in relation_df.iterrows()}

    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    embeddings = np.stack(relation_df["embedding"].values)
    similarities = cosine_similarity(embeddings, query_embedding).flatten()

    seed_indices = np.argsort(similarities)[-topk_seeds:][::-1]
    seed_rows = relation_df.iloc[seed_indices]

    hypernodes = []
    for _, row in seed_rows.iterrows():
        tid = row["_tid"]
        hn = HyperNode(
            triple_ids=[tid],
            path_embedding=row["embedding"].copy(),
            similarity=float(similarities[tid]),
        )
        hypernodes.append(hn)

    # Iterative expansion
    for _ in range(max_hops - 1):
        new_hypernodes = []
        for hn in hypernodes:
            tail_hr_id = _get_tail_human_readable_id(hn, _triple_map)
            if tail_hr_id is None or tail_hr_id not in graph:
                continue
            for neighbor in graph.neighbors(tail_hr_id):
                neighbor_relations = relation_df[
                    (relation_df["head_id"] == tail_hr_id) &
                    (relation_df["tail_id"] == neighbor)
                ]
                for _, rel_row in neighbor_relations.iterrows():
                    tid = rel_row["_tid"]
                    if tid in hn.triple_ids:
                        continue  # avoid cycles
                    new_emb = (hn.path_embedding + rel_row["embedding"]) / 2.0
                    new_sim = float(
                        cosine_similarity(
                            new_emb.reshape(1, -1), query_embedding
                        ).flatten()[0]
                    )
                    new_hypernodes.append(
                        HyperNode(
                            triple_ids=hn.triple_ids + [tid],
                            path_embedding=new_emb,
                            similarity=new_sim,
                        )
                    )
        new_hypernodes.sort(key=lambda x: x.similarity, reverse=True)
        hypernodes.extend(new_hypernodes[:topk_per_hop])

    # Deduplicate by serialized representation
    seen = set()
    unique = []
    for hn in sorted(hypernodes, key=lambda x: x.similarity, reverse=True):
        key = hn.serialize()
        if key not in seen:
            seen.add(key)
            unique.append(hn)

    return unique, _triple_map


def _get_tail_human_readable_id(hn: HyperNode, triple_map: dict):
    """Return the tail entity human_readable_id of the last triple in this HyperNode."""
    row = triple_map[hn.triple_ids[-1]]
    return row["tail_id"]


def compute_hypernode_entity_scores(
    hypernodes: list[HyperNode],
    triple_map: dict,
    entity_df: pd.DataFrame,
) -> dict:
    """Compute per-entity consensus score from hypernodes.

    Returns dict mapping entity index_id → score.
    """
    scores = defaultdict(float)
    hr_to_idx = dict(zip(entity_df["human_readable_id"], entity_df["index_id"]))

    for hn in hypernodes:
        weight = hn.similarity
        entities_in_path = set()
        for tid in hn.triple_ids:
            row = triple_map[tid]
            entities_in_path.add(row["head_id"])
            entities_in_path.add(row["tail_id"])
        if not entities_in_path:
            continue
        contribution = weight / len(entities_in_path)
        for hr_id in entities_in_path:
            idx_id = hr_to_idx.get(hr_id)
            if idx_id is not None:
                scores[idx_id] += contribution

    return dict(scores)


def rerank_by_path_consensus(
    candidates: list,
    entity_scores: dict,
) -> list:
    """Re-rank candidate entities by path consensus score desc."""
    return sorted(candidates, key=lambda e: entity_scores.get(e, 0.0), reverse=True)
