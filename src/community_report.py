import json
import multiprocessing as mp
import os
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from src.llm import llm_invoker
from src.prompts import COMMUNITY_REPORT_PROMPT, COMMUNITY_CONTEXT
from src.utils import *
from src.lm_emb import *
from src.client_reasoning import prep_e_r_content, prep_community_content
from src.triple_text_mapping import (
    make_extractive_community_report,
    select_source_items_for_nodes,
)


def prep_community_report_content(
    level,
    community_entities,
    relationships,
    sub_communities_df=None,
    max_tokens=None,
):
    if (level > 1) and (len(sub_communities_df) > 0):
        community_str_list = prep_community_content(
            sub_communities_df, max_tokens=max_tokens
        )
        community_string = community_str_list[0]
        if len(community_str_list) > 1:
            return community_string
        else:
            remain_token = max_tokens - num_tokens(community_string)
            remain_e_r_list = prep_e_r_content(
                community_entities, relationships, max_tokens=remain_token
            )
            return community_string + remain_e_r_list[0]
    else:
        new_string = prep_e_r_content(
            community_entities, relationships, max_tokens=max_tokens
        )
        e_r_string = new_string[0]
        return e_r_string


def extract_community_report(result, community_id):
    """
    Extract the fields from result if valid.
    """
    required_fields = [
        ("title", str),
        ("summary", str),
        ("findings", list),
        ("rating", float),
        ("rating_explanation", str),
    ]
    # Validate the result
    if dict_has_keys_with_types(result, required_fields):
        return {
            "title": result["title"],
            "summary": result["summary"],
            "findings": result.get("findings"),
            "rating": result.get("rating"),
            "rating_explanation": result.get("rating_explanation"),
        }, True
    # Weak check for title and summary
    elif (
        "title" in result
        and result["title"]
        and "summary" in result
        and result["summary"]
        and "findings" in result
        and result["findings"]
        and "rating" in result
        and result["rating"]
    ):
        return {
            "title": result["title"],
            "summary": result["summary"],
            "findings": result.get("findings"),
            "rating": result.get("rating"),
            "rating_explanation": result.get("rating_explanation"),
        }, True
    else:
        return {
            "title": "CommunityID" + str(community_id),
            "summary": None,
            "findings": None,
            "rating": None,
            "rating_explanation": None,
        }, False


def dict_has_keys_with_types(d, keys_with_types):
    """Check if dict `d` contains keys with expected types as defined in `keys_with_types`."""
    for key, expected_type in keys_with_types:
        if key not in d or not isinstance(d[key], expected_type):
            return False
    return True


def get_adaptive_report_prompt(community_text: str, node_count: int = 0) -> str:
    """Generate a community report prompt with findings count adapted to community size.

    Small communities (2-3 nodes) get 2-3 findings. Large ones (10+) get the
    full 5-10 range. This prevents hallucinations from over-requesting details.
    """
    # Estimate node count from "Entities" table if not provided
    if node_count <= 0:
        # Count lines in the entities section
        entity_section_start = community_text.find("Entities")
        if entity_section_start >= 0:
            entity_lines = community_text[entity_section_start:].strip().split("\n")
            node_count = max(1, len(entity_lines) - 2)  # -2 for header row

    if node_count <= 3:
        findings_range = "2-3"
    elif node_count <= 8:
        findings_range = "3-5"
    else:
        findings_range = "5-10"

    # Replace findings count in the prompt
    import re
    prompt = COMMUNITY_REPORT_PROMPT.format(input_text=community_text)
    prompt = re.sub(
        r"include\s+(?:5-10|3-5|2-3)\s+key\s+findings",
        f"include {findings_range} key findings",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt


def generate_community_report(community_text, args, community_id, max_generate=3):
    node_count = getattr(args, "community_node_count", 0)
    report_prompt = get_adaptive_report_prompt(community_text, node_count=node_count)
    retries = 0
    success = False
    raw_result = None
    extract_result = None
    all_tokens = 0
    
    while not success and retries < max_generate:
        raw_result, cur_token = llm_invoker(
            report_prompt, args, max_tokens=args.max_tokens, json=True
        )
        all_tokens += cur_token
        try:
            output = json.loads(raw_result)

            extract_result, success = extract_community_report(output, community_id)
            if success:
                break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        retries += 1

    if success is False:
        print(f"Failed to generate community report for community:{community_id}")
        if extract_result is None:
            extract_result = {
                "title": "CommunityID" + str(community_id),
                "summary": None,
                "findings": None,
                "rating": None,
                "rating_explanation": None,
            }
        return None, extract_result, all_tokens

    return raw_result, extract_result, all_tokens


def report_embedding(community_report, community_text, args):
    if community_report["title"] is None or community_report["summary"] is None:
        text = community_text
        community_report["title"] = "CommunityID" + str(
            community_report["community_id"]
        )
    else:
        text = community_report["title"] + community_report["summary"]
    embedding = openai_embedding(
        text, args.embedding_api_key, args.embedding_api_base, args.embedding_model
    )
    community_report["embedding"] = embedding
    return community_report


def _validate_report_embeddings(embeddings, expected_dim=None):
    invalid_items = []
    inferred_dim = expected_dim

    for idx, embedding in enumerate(embeddings):
        try:
            embedding_array = np.asarray(embedding, dtype=float)
        except (TypeError, ValueError):
            invalid_items.append((idx, type(embedding).__name__))
            continue

        if embedding_array.ndim != 1 or not np.isfinite(embedding_array).all():
            invalid_items.append((idx, embedding_array.shape))
            continue

        if inferred_dim is None:
            inferred_dim = int(embedding_array.shape[0])
        elif embedding_array.shape[0] != inferred_dim:
            invalid_items.append((idx, embedding_array.shape))

    if invalid_items:
        preview = invalid_items[:5]
        expected_text = (
            f"{inferred_dim}-dim" if inferred_dim is not None else "a consistent"
        )
        raise ValueError(
            f"Invalid community report embeddings: expected {expected_text} finite vectors, "
            f"got {len(invalid_items)} invalid rows, examples={preview}"
        )


def reprot_embedding_batch(community_df, args, num_workers=32):
    def embedding_context(row):
        if row["summary"] is None:
            row["embedding_context"] = row["title"] + row["community_text"]
        else:
            row["embedding_context"] = row["title"] + row["summary"]
        return row

    community_df = community_df.apply(embedding_context, axis=1)

    if args.embedding_local:
        model, tokenizer, device = load_sbert(args.embedding_model_local)

        # 计算 embeddings
        texts = community_df["embedding_context"].tolist()
        community_df["embedding"] = text_to_embedding_batch(
            model, tokenizer, device, texts
        )
    else:

        def get_embedding(row):
            return openai_embedding(
                row["embedding_context"],
                args.embedding_api_key,
                args.embedding_api_base,
                args.embedding_model,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            embeddings = list(
                tqdm(
                    executor.map(
                        lambda row: get_embedding(row),
                        [row for _, row in community_df.iterrows()],
                    ),
                    total=len(community_df),
                    desc="Computing embeddings",
                    unit="community",
                    leave=True,
                )
            )

        _validate_report_embeddings(embeddings)

        # 将结果添加回 DataFrame
        community_df["embedding"] = embeddings

    community_df = community_df.drop(columns=["embedding_context"])
    print("finish compute report batch embedding")
    print(community_df.shape)
    return community_df


def community_report_worker(
    community_id,
    node_list,
    c_c_mapping,
    final_entities,
    final_relationships,
    exist_community_df,
    args,
    level_dict,
    triple_text_mapping,
    chunk_weights,
):

    # 处理单个社区的函数
    print(f"Community {community_id}:")
    community_entities = final_entities.loc[
        final_entities["human_readable_id"].isin(node_list)
    ]

    community_relationships = final_relationships[
        final_relationships["head_id"].isin(node_list)
        | final_relationships["tail_id"].isin(node_list)
    ]

    if len(c_c_mapping) > 0:
        sub_communities = c_c_mapping.get(str(community_id), [])
        sub_communities_df = exist_community_df[
            exist_community_df["community_id"].isin(sub_communities)
        ]
    else:
        sub_communities_df = pd.DataFrame()

    community_level = level_dict.get(community_id, 0)
    source_text_max_tokens = args.source_text_max_tokens or args.max_tokens
    source_items = []
    if args.enable_triple_text_mapping and triple_text_mapping:
        source_items = select_source_items_for_nodes(
            node_list,
            triple_text_mapping,
            chunk_weights,
            top_k=args.source_text_top_k,
            max_tokens=source_text_max_tokens,
        )

    community_text = prep_community_report_content(
        community_level,
        community_entities,
        community_relationships,
        sub_communities_df=sub_communities_df,
        max_tokens=args.max_tokens,
    )

    report_mode = getattr(args, "community_report_mode", "hybrid")
    should_use_extractive = (
        report_mode == "extractive"
        or (
            report_mode == "hybrid"
            and len(community_relationships) >= args.extractive_large_community_threshold
        )
    )

    if should_use_extractive and source_items:
        community_report, community_text = make_extractive_community_report(
            community_id,
            community_level,
            node_list,
            source_items,
        )
        raw_result = None
        all_tokens = 0
    else:
        raw_result, community_report, all_tokens = generate_community_report(
            community_text, args, community_id
        )
        community_report["source_type"] = "llm_report"
        community_report["source_texts"] = [item["text"] for item in source_items]
        community_report["source_text_unit_ids"] = [
            item["text_unit_id"] for item in source_items
        ]
        community_report["source_relationship_ids"] = [
            item["relationship_id"] for item in source_items
        ]

    community_report["community_id"] = community_id
    community_report["level"] = level_dict.get(
        community_id, None
    )  # 使用 level_dict 获取对应的 level
    community_report["community_nodes"] = node_list
    community_report["raw_result"] = raw_result
    community_report["community_text"] = community_text

    if not community_report.get("title"):
        community_report["title"] = f"CommunityID{community_id}"

    return community_report, all_tokens


def community_report_batch(
    communities: dict[str, list[str]],
    c_c_mapping: dict[str, list[str]],
    final_entities,
    final_relationships,
    exist_community_df,
    args,
    error_save_path,
    level_dict: dict[str, int],
    triple_text_mapping=None,
    chunk_weights=None,
):
    results_community = []
    total_tokens = 0
    
    # 创建进程池
    with mp.Pool(processes=args.num_workers) as pool:

        # 使用 partial 来将固定的参数传递到 worker 中
        process_func = partial(
            community_report_worker,
            c_c_mapping=c_c_mapping,
            final_entities=final_entities,
            final_relationships=final_relationships,
            exist_community_df=exist_community_df,
            args=args,
            level_dict=level_dict,
            triple_text_mapping=triple_text_mapping or {},
            chunk_weights=chunk_weights or {},
        )

        # 并行处理每个社区
        try:
            results = pool.starmap(process_func, communities.items())
            for community_report, all_tokens in results:
                results_community.append(community_report)
                total_tokens += all_tokens
        except Exception as e:
            logging.error(f"Error processing communities: {e}")
            # 保存已成功的结果
            if results_community:
                community_df = pd.DataFrame(results_community)
                community_df.to_csv(error_save_path, index=False)  # 保存到文件
            raise  # 重新抛出异常以终止处理

    community_df = pd.DataFrame(results_community)
    if "source_type" in community_df.columns:
        source_counts = community_df["source_type"].fillna("unknown").value_counts()
        print(f"Community report source types: {source_counts.to_dict()}")
    community_df = reprot_embedding_batch(
        community_df, args, num_workers=args.embedding_num_workers
    )

    return community_df, total_tokens


def community_report_for_level(
    results_by_level, args, final_entities, final_relationships
):
    results_community = []
    all_tokens = 0
    for level, communities in results_by_level.items():
        print(f"Create community report for level: {level} ")
        print(f"Number of communities in this level: {len(communities)}")
        # 构建一个 level_dict，key 为 community_id，value 为 level
        level_dict = {community_id: level for community_id in communities.keys()}

        res, cur_token = community_report_batch(
            communities=communities,
            final_entities=final_entities,
            final_relationships=final_relationships,
            args=args,
            level_dict=level_dict,
        )
        all_tokens += cur_token

        results_community.append(res)  # 将 DataFrame 添加到列表中

    # 使用 pd.concat 合并所有 DataFrame
    community_df = pd.concat(results_community, ignore_index=True)
    return community_df, all_tokens


def confirm_community_result(results_by_level):
    levels = sorted(
        results_by_level.keys(), reverse=True
    )  # Start from the highest level
    lower_to_higher_assignments = (
        {}
    )  # To track the assignment of lower-level communities

    for i in range(len(levels) - 1):  # Compare each level with the next lower level
        higher_level = levels[i]
        lower_level = levels[i + 1]
        print(f"Checking level {higher_level} against level {lower_level}...")

        higher_level_communities = results_by_level[higher_level]
        lower_level_communities = results_by_level[lower_level]

        # Track which lower-level communities have already been assigned to a higher-level community
        lower_community_to_higher = {}
        for higher_community_id, higher_node_list in higher_level_communities.items():
            higher_node_set = set(higher_node_list)

            # Check each node in the higher-level community against the lower-level communities
            for lower_community_id, lower_node_list in lower_level_communities.items():
                lower_node_set = set(lower_node_list)

                # If any nodes from the lower-level community are in the higher-level community
                if higher_node_set.intersection(lower_node_set):
                    if lower_community_id in lower_community_to_higher:
                        # If this lower-level community is already assigned to a different higher-level community
                        assigned_higher_community = lower_community_to_higher[
                            lower_community_id
                        ]
                        if assigned_higher_community != higher_community_id:
                            print(
                                f"Error: Lower-level community {lower_community_id} is part of both "
                                f"higher-level community {higher_community_id} and {assigned_higher_community}!"
                            )
                    else:
                        # Assign this lower-level community to the current higher-level community
                        lower_community_to_higher[lower_community_id] = (
                            higher_community_id
                        )

                        # Store the assignment in the tracking dictionary
                        if lower_community_id not in lower_to_higher_assignments:
                            lower_to_higher_assignments[lower_community_id] = []
                        lower_to_higher_assignments[lower_community_id].append(
                            higher_community_id
                        )

        print(f"Finished checking level {higher_level} against level {lower_level}.\n")

    # Output the assignment results
    print("Assignment of lower-level communities to higher-level communities:")
    for lower_community_id, higher_communities in lower_to_higher_assignments.items():
        print(
            f"Lower-level community {lower_community_id} is assigned to higher-level communities: {higher_communities}"
        )

        # Optionally, output the contents (nodes) of the assigned communities
        for higher_level in higher_communities:
            print(
                f"  Higher-level community {higher_level} content: {results_by_level[levels[i]][higher_level]}"
            )
        print()

    print("Finished checking all levels.")


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    cos_graph = compute_distance(graph=graph)
    # results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)

    # confirm_community_result(results_by_level)

    # community_df = community_report_for_level(
    #     results_by_level, args, final_entities, final_relationships
    # )

    # output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    # community_df.to_csv(output_path, index=False)
