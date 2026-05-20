"""Query an existing expanded ArchRAG index with the real API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def latest_artifacts() -> Path:
    output_dir = ROOT / "corpus" / "output"
    runs = sorted([p for p in output_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No GraphRAG output runs found under corpus/output")
    return runs[0] / "artifacts"


def build_args(artifacts: Path, index_dir: Path, qa_path: Path) -> SimpleNamespace:
    settings = yaml.safe_load((ROOT / "corpus" / "settings.yaml").read_text(encoding="utf-8"))
    llm = settings["llm"]
    emb = settings["embeddings"]["llm"]
    return SimpleNamespace(
        project="expanded_archrag_real",
        base_path=str(artifacts),
        relationship_filename="create_final_relationships.parquet",
        entity_filename="create_final_entities.parquet",
        output_dir=str(index_dir),
        api_key=llm["api_key"],
        api_base=llm["api_base"],
        engine=llm["model"],
        max_tokens=3000,
        temperature=0.1,
        max_retries=3,
        embedding_model=emb["model"],
        embedding_api_key=emb["api_key"],
        embedding_api_base=emb["api_base"],
        dataset_name="expanded_demo",
        dataset_path=str(qa_path),
        doc_idx=-1,
        eval_mode="DocQA",
        only_entity=False,
        wo_hierarchical=True,
        ppr_refine=False,
        topk_e=10,
        topk_chunk=0,
        num_workers=1,
        print_log=True,
        debug_flag=False,
        disable_wandb=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "expanded_archrag_real_index")
    parser.add_argument("--qa-path", type=Path, default=ROOT / "corpus" / "expanded_demo_qa.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--generate-strategy", choices=["direct", "mr"], default="direct")
    parser.add_argument("--k-each-level", type=int, default=5)
    parser.add_argument("--k-final", type=int, default=10)
    parser.add_argument("--topk-e", type=int, default=10)
    args_cli = parser.parse_args()

    from src.inference import hcarag, load_index

    artifacts = args_cli.artifacts or latest_artifacts()
    args = build_args(artifacts, args_cli.index_dir, args_cli.qa_path)
    index_dict = load_index(args)

    query_paras = {
        "strategy": "global",
        "only_entity": args.only_entity,
        "wo_hierarchical": args.wo_hierarchical,
        "k_each_level": args_cli.k_each_level,
        "k_final": args_cli.k_final,
        "topk_e": args_cli.topk_e,
        "all_k_inference": 10,
        "ppr_refine": args.ppr_refine,
        "generate_strategy": args_cli.generate_strategy,
        "response_type": "QA",
        "involve_llm_res": args_cli.generate_strategy == "mr",
        "topk_chunk": args.topk_chunk,
        "range_level": 2,
    }

    qa_df = pd.read_json(args.dataset_path, orient="records", lines=True).head(args_cli.limit)
    rows = []
    for _, row in qa_df.iterrows():
        response, total_token = hcarag(row["question"], index_dict, query_paras, args)
        rows.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "pred": response["pred"],
                "raw_result": response["raw_result"],
                "total_token": total_token,
                "topk_entity": response["topk_entity"],
                "topk_community": response["topk_community"],
                "topk_related_r": response["topk_related_r"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_path = args_cli.index_dir / f"expanded_real_qa_results_{args_cli.generate_strategy}_k{args_cli.k_each_level}.csv"
    result_df.to_csv(result_path, index=False)
    print(f"QA results: {result_path}")
    for row in rows:
        print(f"Q: {row['question']}")
        print(f"Pred: {row['pred']}")


if __name__ == "__main__":
    main()
