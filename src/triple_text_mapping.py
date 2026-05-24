import ast
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import num_tokens


TRIPLE_TEXT_MAPPING_FILENAME = "triple_text_mapping.pkl"
CHUNK_WEIGHTS_FILENAME = "chunk_weights.pkl"
COMMUNITY_SOURCE_TEXT_FILENAME = "community_source_text.csv"


def parse_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if not isinstance(value, str):
        return [str(value)]

    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if item is not None]
        if parsed is None:
            return []
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [item.strip() for item in value.split(",") if item.strip()]


def read_text_units(
    file_path: str,
    text_unit_filename: str = "create_final_text_units.parquet",
) -> pd.DataFrame | None:
    data_path = Path(file_path)
    candidates = [data_path / text_unit_filename]

    stem = Path(text_unit_filename).stem
    candidates.extend(
        [
            data_path / f"{stem}.parquet",
            data_path / f"{stem}.csv",
            data_path / "create_final_text_units.parquet",
            data_path / "create_final_text_units.csv",
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        if candidate.suffix == ".csv":
            return pd.read_csv(candidate)
        return pd.read_parquet(candidate)
    return None


def build_triple_text_mapping(
    relationships_df: pd.DataFrame,
    text_units_df: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    if text_units_df is None or text_units_df.empty:
        return {}, {}

    text_by_id = dict(zip(text_units_df["id"].astype(str), text_units_df["text"]))
    token_by_id = (
        dict(zip(text_units_df["id"].astype(str), text_units_df["n_tokens"]))
        if "n_tokens" in text_units_df.columns
        else {}
    )

    mapping: dict[str, dict[str, Any]] = {}
    chunk_weights: dict[str, int] = {}

    for row_idx, row in relationships_df.iterrows():
        triple_id = str(row.get("id") or row.get("human_readable_id") or row_idx)
        text_unit_ids = parse_id_list(row.get("text_unit_ids"))
        source_chunks = [chunk_id for chunk_id in text_unit_ids if chunk_id in text_by_id]
        chunk_texts = [str(text_by_id[chunk_id]) for chunk_id in source_chunks]

        for chunk_id in source_chunks:
            chunk_weights[chunk_id] = chunk_weights.get(chunk_id, 0) + 1

        mapping[triple_id] = {
            "relationship_id": triple_id,
            "human_readable_id": row.get("human_readable_id"),
            "description": row.get("description", ""),
            "source": row.get("source"),
            "target": row.get("target"),
            "head": row.get("head_id"),
            "tail": row.get("tail_id"),
            "source_index_id": row.get("source_index_id"),
            "target_index_id": row.get("target_index_id"),
            "text_unit_ids": text_unit_ids,
            "source_chunks": source_chunks,
            "chunk_texts": chunk_texts,
            "chunk_tokens": [token_by_id.get(chunk_id) for chunk_id in source_chunks],
        }

    return mapping, chunk_weights


def save_triple_text_artifacts(
    output_dir: str,
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, TRIPLE_TEXT_MAPPING_FILENAME), "wb") as f:
        pickle.dump(triple_text_mapping, f)
    with open(os.path.join(output_dir, CHUNK_WEIGHTS_FILENAME), "wb") as f:
        pickle.dump(chunk_weights, f)


def load_triple_text_artifacts(
    output_dir: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    mapping_path = os.path.join(output_dir, TRIPLE_TEXT_MAPPING_FILENAME)
    weights_path = os.path.join(output_dir, CHUNK_WEIGHTS_FILENAME)
    if not os.path.exists(mapping_path) or not os.path.exists(weights_path):
        return {}, {}
    with open(mapping_path, "rb") as f:
        triple_text_mapping = pickle.load(f)
    with open(weights_path, "rb") as f:
        chunk_weights = pickle.load(f)
    return triple_text_mapping, chunk_weights


def _normalize_nodes(nodes: Any) -> set[Any]:
    if isinstance(nodes, str):
        nodes = parse_id_list(nodes)
    if nodes is None:
        return set()
    return set(nodes)


def _same_node(left: Any, right_set: set[Any]) -> bool:
    return left in right_set or str(left) in {str(item) for item in right_set}


def _dedupe_source_items(source_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_chunks = set()
    deduped = []
    for item in source_items:
        chunk_id = item["text_unit_id"]
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        deduped.append(item)
    return deduped


def select_source_items_for_nodes(
    nodes: Any,
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
    top_k: int = 3,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    node_set = _normalize_nodes(nodes)
    source_items = []

    for triple_id, info in triple_text_mapping.items():
        if not (_same_node(info.get("head"), node_set) or _same_node(info.get("tail"), node_set)):
            continue
        for chunk_id, chunk_text in zip(info.get("source_chunks", []), info.get("chunk_texts", [])):
            source_items.append(
                {
                    "relationship_id": triple_id,
                    "text_unit_id": chunk_id,
                    "text": chunk_text,
                    "weight": chunk_weights.get(chunk_id, 0),
                    "description": info.get("description", ""),
                }
            )

    source_items.sort(key=lambda item: (item["weight"], item["relationship_id"]), reverse=True)
    source_items = _dedupe_source_items(source_items)
    if top_k:
        source_items = source_items[:top_k]

    if max_tokens:
        selected = []
        current_tokens = 0
        for item in source_items:
            item_tokens = num_tokens(item["text"])
            if selected and current_tokens + item_tokens > max_tokens:
                break
            selected.append(item)
            current_tokens += item_tokens
        source_items = selected

    return source_items


def get_relationship_source_items(
    relation_ids: list[Any],
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
    top_k: int = 3,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    source_items = []
    for relation_id in relation_ids:
        info = triple_text_mapping.get(str(relation_id))
        if not info:
            continue
        for chunk_id, chunk_text in zip(info.get("source_chunks", []), info.get("chunk_texts", [])):
            source_items.append(
                {
                    "relationship_id": str(relation_id),
                    "text_unit_id": chunk_id,
                    "text": chunk_text,
                    "weight": chunk_weights.get(chunk_id, 0),
                    "description": info.get("description", ""),
                }
            )

    source_items.sort(key=lambda item: (item["weight"], item["relationship_id"]), reverse=True)
    source_items = _dedupe_source_items(source_items)
    if top_k:
        source_items = source_items[:top_k]

    if max_tokens:
        selected = []
        current_tokens = 0
        for item in source_items:
            item_tokens = num_tokens(item["text"])
            if selected and current_tokens + item_tokens > max_tokens:
                break
            selected.append(item)
            current_tokens += item_tokens
        source_items = selected
    return source_items


def make_extractive_community_report(
    community_id: Any,
    level: int,
    node_list: list[Any],
    source_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    evidence_lines = []
    for item in source_items:
        evidence_lines.append(
            f"[TextUnit {item['text_unit_id']} | Relationship {item['relationship_id']}] {item['text']}"
        )
    evidence_text = "\n".join(evidence_lines)
    summary = "Source evidence:\n" + evidence_text if evidence_text else ""

    report = {
        "title": f"Community {community_id} Source Evidence",
        "summary": summary,
        "findings": [
            {
                "summary": "Extractive source evidence",
                "explanation": (
                    "This community representation is assembled directly from "
                    "GraphRAG text units linked to its relationships."
                ),
            }
        ],
        "rating": 5.0,
        "rating_explanation": "Direct source text evidence; no LLM community report was generated.",
        "source_type": "direct_text",
        "source_texts": [item["text"] for item in source_items],
        "source_text_unit_ids": [item["text_unit_id"] for item in source_items],
        "source_relationship_ids": [item["relationship_id"] for item in source_items],
        "community_id": community_id,
        "level": level,
        "community_nodes": node_list,
        "raw_result": None,
        "community_text": summary,
    }
    return report, summary


def enrich_community_with_source_text(
    community_df: pd.DataFrame,
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
    top_k: int = 3,
    max_tokens: int | None = None,
) -> pd.DataFrame:
    enriched = community_df.copy()
    if enriched.empty or not triple_text_mapping:
        return enriched

    for column in ["source_texts", "source_text_unit_ids", "source_relationship_ids"]:
        if column not in enriched.columns:
            enriched[column] = None

    for idx, row in enriched.iterrows():
        source_items = select_source_items_for_nodes(
            row.get("community_nodes"),
            triple_text_mapping,
            chunk_weights,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        enriched.at[idx, "source_texts"] = [item["text"] for item in source_items]
        enriched.at[idx, "source_text_unit_ids"] = [item["text_unit_id"] for item in source_items]
        enriched.at[idx, "source_relationship_ids"] = [
            item["relationship_id"] for item in source_items
        ]
    return enriched


def enrich_relationships_with_source_text(
    relation_df: pd.DataFrame,
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
    top_k: int = 1,
    max_tokens: int | None = None,
) -> pd.DataFrame:
    enriched = relation_df.copy()
    if enriched.empty or not triple_text_mapping:
        return enriched

    for column in ["source_texts", "source_text_unit_ids"]:
        if column not in enriched.columns:
            enriched[column] = None

    for idx, row in enriched.iterrows():
        relation_id = row.get("id") or row.get("human_readable_id") or idx
        source_items = get_relationship_source_items(
            [relation_id],
            triple_text_mapping,
            chunk_weights,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        enriched.at[idx, "source_texts"] = [item["text"] for item in source_items]
        enriched.at[idx, "source_text_unit_ids"] = [item["text_unit_id"] for item in source_items]
    return enriched


def write_community_source_table(
    community_df: pd.DataFrame,
    output_dir: str,
    triple_text_mapping: dict[str, dict[str, Any]],
    chunk_weights: dict[str, int],
    top_k: int = 3,
    max_tokens: int | None = None,
) -> None:
    if community_df.empty or not triple_text_mapping:
        return

    rows = []
    for _, row in community_df.iterrows():
        source_items = select_source_items_for_nodes(
            row.get("community_nodes"),
            triple_text_mapping,
            chunk_weights,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        rows.append(
            {
                "community_id": row.get("community_id"),
                "level": row.get("level"),
                "source_text_unit_ids": [item["text_unit_id"] for item in source_items],
                "source_relationship_ids": [item["relationship_id"] for item in source_items],
                "source_texts": [item["text"] for item in source_items],
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, COMMUNITY_SOURCE_TEXT_FILENAME),
        index=False,
    )
