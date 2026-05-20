"""Run a tiny end-to-end ArchRAG demo without external APIs.

This script creates a small corpus, a GraphRAG-like entity/relationship graph,
builds the ArchRAG hierarchical HCHNSW index, and answers a few questions.
It monkeypatches the project LLM and embedding calls with deterministic local
functions so the demo is cheap and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


DIM = 32
KEYWORDS = {
    "apollo": 0,
    "moon": 1,
    "armstrong": 2,
    "aldrin": 3,
    "collins": 4,
    "1969": 5,
    "everest": 6,
    "himalayas": 7,
    "nepal": 8,
    "china": 9,
    "hillary": 10,
    "tenzing": 11,
    "1953": 12,
    "python": 13,
    "guido": 14,
    "readability": 15,
    "package": 16,
    "pip": 17,
}


def local_embedding(text: str, *_args, **_kwargs) -> list[float]:
    """Small deterministic lexical embedding used only for this demo."""
    text = (text or "").lower()
    vec = np.zeros(DIM, dtype=np.float32)
    for keyword, idx in KEYWORDS.items():
        if keyword in text:
            vec[idx] += 3.0

    for token in re.findall(r"[a-z0-9]+", text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vec[18 + digest[0] % (DIM - 18)] += 0.15

    norm = float(np.linalg.norm(vec))
    if norm == 0:
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).astype(float).tolist()


def _answer_for_question(prompt: str) -> str:
    prompt_l = prompt.lower()
    if "walked on the lunar surface" in prompt_l or "who walked" in prompt_l:
        answer = "Neil Armstrong | Buzz Aldrin"
        analysis = "The Apollo 11 facts identify Neil Armstrong and Buzz Aldrin as the astronauts who walked on the lunar surface."
    elif "where is mount everest" in prompt_l or "everest located" in prompt_l:
        answer = "The Himalayas, on the border between Nepal and China"
        analysis = "The Everest facts state that the mountain is in the Himalayas and lies on the Nepal-China border."
    elif "first confirmed summit" in prompt_l or "summit" in prompt_l:
        answer = "Edmund Hillary | Tenzing Norgay"
        analysis = "The Everest facts describe the first confirmed summit in 1953 by Edmund Hillary and Tenzing Norgay."
    elif "created python" in prompt_l or "who created python" in prompt_l:
        answer = "Guido van Rossum"
        analysis = "The Python facts say that Guido van Rossum created Python."
    elif "package installer" in prompt_l or "what tool" in prompt_l:
        answer = "pip"
        analysis = "The Python facts identify pip as a package installer commonly used with Python."
    else:
        answer = "No answer found"
        analysis = "The tiny demo mock did not recognize this question."
    return f"Direct Answer\n{answer}\nBrief Analysis\n{analysis}"


def local_llm(input_text: str, args, temperature=0.7, max_tokens=4000, max_retries=5, json=False):
    """Deterministic LLM stand-in matching the project parsers."""
    text_l = input_text.lower()

    if json and '"points"' in input_text:
        point = _answer_for_question(input_text).replace("Direct Answer\n", "").replace("\nBrief Analysis\n", ". ")
        return json_dumps({"points": [{"description": point, "score": 100}]}), 64

    if json and '"finds"' in input_text:
        finds = []
        max_level_match = re.search(r"Max level:\s*(\d+)", input_text)
        max_level = int(max_level_match.group(1)) if max_level_match else 1
        for level in range(1, max_level + 1):
            finds.append(
                {
                    "id": level,
                    "rate": 8.0 if level == 1 else 6.0,
                    "rating_explanation": "Relevant demo hierarchy level.",
                }
            )
        return json_dumps({"finds": finds}), 64

    if json and '"summary"' in input_text and '"rate"' in input_text and "level report" in text_l:
        return json_dumps(
            {
                "summary": "This demo level groups related facts into compact topical communities.",
                "rate": 7.0,
                "rating_explanation": "The level is useful for routing questions to the right topic.",
            }
        ), 64

    if json and '"title"' in input_text and '"findings"' in input_text:
        names = re.findall(r"^\s*\d+\s*,\s*([^,\n]+)\s*,", input_text, flags=re.MULTILINE)
        clean_names = [name.strip() for name in names if name.strip()][:4]
        if not clean_names:
            clean_names = ["Demo Community"]
        title = " and ".join(clean_names[:2])
        summary = f"This community contains facts about {', '.join(clean_names)}."
        return json_dumps(
            {
                "title": title,
                "summary": summary,
                "rating": 5.0,
                "rating_explanation": "Small demo community with directly connected facts.",
                "findings": [
                    {
                        "summary": "Related demo facts",
                        "explanation": summary,
                    }
                ],
            }
        ), 96

    return _answer_for_question(input_text), 96


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def patch_runtime() -> None:
    import src.client_reasoning as client_reasoning
    import src.community_report as community_report
    import src.inference as inference
    import src.lm_emb as lm_emb
    import src.llm as llm
    import src.utils as utils

    lm_emb.openai_embedding = local_embedding
    utils.openai_embedding = local_embedding
    community_report.openai_embedding = local_embedding
    inference.openai_embedding = local_embedding

    llm.llm_invoker = local_llm
    community_report.llm_invoker = local_llm
    client_reasoning.llm_invoker = local_llm
    inference.llm_invoker = local_llm


def reset_tiny_corpus() -> Path:
    corpus_dir = ROOT / "corpus"
    input_dir = corpus_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    docs = {
        "doc_0001.txt": (
            "Apollo 11 was the first crewed mission to land on the Moon.\n"
            "The mission launched in July 1969.\n"
            "Neil Armstrong and Buzz Aldrin walked on the lunar surface.\n"
            "Michael Collins stayed in lunar orbit in the command module.\n"
        ),
        "doc_0002.txt": (
            "Mount Everest is the highest mountain above sea level.\n"
            "Everest is located in the Himalayas on the border between Nepal and China.\n"
            "The first confirmed summit was in 1953 by Edmund Hillary and Tenzing Norgay.\n"
        ),
        "doc_0003.txt": (
            "Python is a programming language created by Guido van Rossum.\n"
            "Python emphasizes readability and has a large package ecosystem.\n"
            "pip is a package installer commonly used with Python projects.\n"
        ),
    }
    for name, content in docs.items():
        (input_dir / name).write_text(content, encoding="utf-8")

    qa_path = corpus_dir / "tiny_demo_qa.jsonl"
    qa_rows = [
        {"question": "Who walked on the lunar surface during Apollo 11?", "answer": "Neil Armstrong|Buzz Aldrin"},
        {"question": "Where is Mount Everest located?", "answer": "Himalayas|Nepal|China"},
        {"question": "Who created Python?", "answer": "Guido van Rossum"},
    ]
    qa_path.write_text("\n".join(json_dumps(row) for row in qa_rows) + "\n", encoding="utf-8")
    return qa_path


def write_tiny_graph(graph_dir: Path) -> None:
    if graph_dir.exists():
        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    entities = [
        (0, "APOLLO 11", "Apollo 11 was the first crewed mission to land on the Moon."),
        (1, "MOON", "The Moon was the destination of Apollo 11."),
        (2, "NEIL ARMSTRONG", "Neil Armstrong walked on the lunar surface during Apollo 11."),
        (3, "BUZZ ALDRIN", "Buzz Aldrin walked on the lunar surface during Apollo 11."),
        (4, "MICHAEL COLLINS", "Michael Collins stayed in lunar orbit in the command module."),
        (5, "MOUNT EVEREST", "Mount Everest is the highest mountain above sea level."),
        (6, "HIMALAYAS", "The Himalayas contain Mount Everest."),
        (7, "NEPAL", "Nepal borders the Mount Everest region."),
        (8, "CHINA", "China borders the Mount Everest region."),
        (9, "EDMUND HILLARY", "Edmund Hillary made the first confirmed Everest summit in 1953."),
        (10, "TENZING NORGAY", "Tenzing Norgay made the first confirmed Everest summit in 1953."),
        (11, "PYTHON", "Python is a programming language created by Guido van Rossum."),
        (12, "GUIDO VAN ROSSUM", "Guido van Rossum created Python."),
        (13, "PIP", "pip is a package installer commonly used with Python projects."),
    ]
    entity_df = pd.DataFrame(
        [
            {
                "id": str(hid),
                "human_readable_id": hid,
                "name": name,
                "type": "concept",
                "description": desc,
                "text_unit_ids": [],
                "description_embedding": local_embedding(f"{name} {desc}"),
            }
            for hid, name, desc in entities
        ]
    )

    edges = [
        (0, "APOLLO 11", "MOON", 0, 1, "Apollo 11 landed on the Moon."),
        (1, "APOLLO 11", "NEIL ARMSTRONG", 0, 2, "Neil Armstrong was an Apollo 11 astronaut."),
        (2, "APOLLO 11", "BUZZ ALDRIN", 0, 3, "Buzz Aldrin was an Apollo 11 astronaut."),
        (3, "APOLLO 11", "MICHAEL COLLINS", 0, 4, "Michael Collins remained in lunar orbit during Apollo 11."),
        (4, "MOUNT EVEREST", "HIMALAYAS", 5, 6, "Mount Everest is located in the Himalayas."),
        (5, "MOUNT EVEREST", "NEPAL", 5, 7, "Mount Everest lies on the Nepal border."),
        (6, "MOUNT EVEREST", "CHINA", 5, 8, "Mount Everest lies on the China border."),
        (7, "MOUNT EVEREST", "EDMUND HILLARY", 5, 9, "Edmund Hillary made the first confirmed Everest summit."),
        (8, "MOUNT EVEREST", "TENZING NORGAY", 5, 10, "Tenzing Norgay made the first confirmed Everest summit."),
        (9, "PYTHON", "GUIDO VAN ROSSUM", 11, 12, "Guido van Rossum created Python."),
        (10, "PYTHON", "PIP", 11, 13, "pip is a package installer used with Python."),
    ]
    degree = {}
    for *_prefix, head_id, tail_id, _desc in edges:
        degree[head_id] = degree.get(head_id, 0) + 1
        degree[tail_id] = degree.get(tail_id, 0) + 1

    relationship_df = pd.DataFrame(
        [
            {
                "id": str(rid),
                "human_readable_id": rid,
                "source": source,
                "target": target,
                "head_id": head_id,
                "tail_id": tail_id,
                "description": desc,
                "weight": 1.0,
                "source_degree": degree[head_id],
                "target_degree": degree[tail_id],
                "rank": degree[head_id] + degree[tail_id],
                "text_unit_ids": [],
                "relation_embedding": local_embedding(desc),
            }
            for rid, source, target, head_id, tail_id, desc in edges
        ]
    )

    entity_df.to_parquet(graph_dir / "create_final_entities.parquet", index=False)
    relationship_df.to_parquet(graph_dir / "create_final_relationships.parquet", index=False)


def build_args(graph_dir: Path, index_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project="tiny_archrag_demo",
        base_path=str(graph_dir),
        relationship_filename="create_final_relationships.parquet",
        entity_filename="create_final_entities.parquet",
        output_dir=str(index_dir),
        wx_weight=0.7,
        search_k=2.0,
        m_du_scale=1,
        seed=0xDEADBEEF,
        max_level=2,
        min_clusters=1,
        max_cluster_size=4,
        augment_graph=True,
        cluster_method="weighted_leiden",
        api_key="local-demo",
        api_base="local-demo",
        engine="local-demo",
        max_tokens=2500,
        temperature=0.1,
        max_community_tokens=2500,
        max_retries=2,
        embedding_local=False,
        embedding_model_local="local-demo",
        embedding_model="local-demo",
        embedding_api_key="local-demo",
        embedding_api_base="local-demo",
        embedding_num_workers=1,
        entity_second_embedding=True,
        num_workers=1,
        print_log=True,
        debug_flag=False,
        dataset_name="tiny_demo",
        dataset_path=str(ROOT / "corpus" / "tiny_demo_qa.jsonl"),
        doc_idx=-1,
        eval_mode="DocQA",
        strategy="global",
        k_each_level=4,
        k_final=8,
        topk_e=6,
        all_k_inference=8,
        only_entity=False,
        wo_hierarchical=False,
        ppr_refine=False,
        involve_llm_res=True,
        topk_chunk=0,
        generate_strategy="mr",
        response_type="QA",
        range_level=2,
        disable_wandb=True,
    )


def run_demo(rebuild: bool) -> None:
    patch_runtime()

    graph_dir = ROOT / "tiny_archrag_graph"
    index_dir = ROOT / "tiny_archrag_index"
    qa_path = reset_tiny_corpus()
    write_tiny_graph(graph_dir)

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    from src.index import make_hc_index
    from src.inference import hcarag, load_index

    args = build_args(graph_dir, index_dir)
    make_hc_index(args)
    index_dict = load_index(args)

    query_paras = {
        "strategy": args.strategy,
        "only_entity": args.only_entity,
        "wo_hierarchical": args.wo_hierarchical,
        "k_each_level": args.k_each_level,
        "k_final": args.k_final,
        "topk_e": args.topk_e,
        "all_k_inference": args.all_k_inference,
        "ppr_refine": args.ppr_refine,
        "generate_strategy": args.generate_strategy,
        "response_type": args.response_type,
        "involve_llm_res": args.involve_llm_res,
        "topk_chunk": args.topk_chunk,
        "range_level": args.range_level,
    }

    qa_df = pd.read_json(qa_path, lines=True)
    rows = []
    for _, row in qa_df.iterrows():
        response, total_token = hcarag(row["question"], index_dict, query_paras, args)
        rows.append(
            {
                "question": row["question"],
                "gold": row["answer"],
                "pred": response["pred"],
                "tokens_mocked": total_token,
                "topk_entity": response["topk_entity"],
                "topk_community": response["topk_community"],
                "topk_related_r": response["topk_related_r"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_path = index_dir / "tiny_demo_results.csv"
    result_df.to_csv(result_path, index=False)

    print("\n=== Tiny ArchRAG demo finished ===")
    print(f"Corpus input: {ROOT / 'corpus' / 'input'}")
    print(f"Graph files: {graph_dir}")
    print(f"Index files: {index_dir}")
    print(f"QA result: {result_path}")
    print("\nPredictions:")
    for row in rows:
        print(f"- Q: {row['question']}")
        print(f"  A: {row['pred']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rebuild", action="store_true", help="Keep an existing tiny_archrag_index directory.")
    args = parser.parse_args()
    run_demo(rebuild=not args.no_rebuild)


if __name__ == "__main__":
    main()
