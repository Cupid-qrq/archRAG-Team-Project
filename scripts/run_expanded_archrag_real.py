"""Build and query ArchRAG for the expanded Aurora Harbor demo corpus."""

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
        wx_weight=0.75,
        search_k=2.0,
        m_du_scale=1,
        seed=0xDEADBEEF,
        max_level=3,
        min_clusters=2,
        max_cluster_size=8,
        augment_graph=True,
        cluster_method="weighted_leiden",
        api_key=llm["api_key"],
        api_base=llm["api_base"],
        engine=llm["model"],
        max_tokens=3000,
        temperature=0.1,
        max_community_tokens=3000,
        max_retries=3,
        embedding_local=False,
        embedding_model_local="",
        embedding_model=emb["model"],
        embedding_api_key=emb["api_key"],
        embedding_api_base=emb["api_base"],
        embedding_num_workers=4,
        entity_second_embedding=True,
        num_workers=4,
        print_log=True,
        debug_flag=False,
        dataset_name="expanded_demo",
        dataset_path=str(qa_path),
        doc_idx=-1,
        eval_mode="DocQA",
        strategy="global",
        k_each_level=6,
        k_final=12,
        topk_e=10,
        all_k_inference=12,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "expanded_archrag_real_index")
    parser.add_argument("--qa-path", type=Path, default=ROOT / "corpus" / "expanded_demo_qa.jsonl")
    args_cli = parser.parse_args()

    artifacts = args_cli.artifacts or latest_artifacts()
    index_dir = args_cli.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    from src.index import make_hc_index
    from src.inference import hcarag, load_index

    args = build_args(artifacts, index_dir, args_cli.qa_path)
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

    qa_df = pd.read_json(args.dataset_path, orient="records", lines=True)
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
    result_path = index_dir / "expanded_real_qa_results.csv"
    result_df.to_csv(result_path, index=False)
    print("\n=== Expanded real ArchRAG flow finished ===")
    print(f"GraphRAG artifacts: {artifacts}")
    print(f"ArchRAG index: {index_dir}")
    print(f"QA results: {result_path}")
    for row in rows:
        print(f"Q: {row['question']}")
        print(f"Pred: {row['pred']}")


if __name__ == "__main__":
    main()
