base_path="archrag" # input dataset path 
relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"
text_unit_filename="create_final_text_units.parquet"
output_dir="archrag_index" 
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=10
max_cluster_size=15
entity_second_embedding=True
api_key="" #TODO
api_base="" #TODO
engine="gpt-5.5" # llm engine

# If you have embedding model settings, uncomment and set them here
# And make sure to uncomment the corresponding line in the nohup command below
# --embedding_model $embedding_model --embedding_api_key $embedding_api_key --embedding_api_base $embedding_api_base \    
embedding_model="" # TODO
embedding_api_key="dummy"
embedding_api_base="" # TODO

augment_graph=True
cluster_method="weighted_leiden"
enable_triple_text_mapping=True
community_report_mode="hybrid"
extractive_large_community_threshold=5
source_text_top_k=3
source_text_max_tokens=0
num_workers=10

log_file="./index.log"
python_file="src/index.py"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7

nohup python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
    --entity_filename $entity_filename --text_unit_filename $text_unit_filename \
    --output_dir $output_dir --wx_weight $wx_weight --m_du_scale $m_du_scale --max_level $max_level \
    --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
    --entity_second_embedding $entity_second_embedding \
    --engine $engine --num_workers $num_workers \
    --augment_graph $augment_graph --cluster_method $cluster_method \
    --enable_triple_text_mapping $enable_triple_text_mapping --community_report_mode $community_report_mode \
    --extractive_large_community_threshold $extractive_large_community_threshold \
    --source_text_top_k $source_text_top_k --source_text_max_tokens $source_text_max_tokens \
    --embedding_model $embedding_model --embedding_api_key $embedding_api_key --embedding_api_base $embedding_api_base \
    --api_key $api_key --api_base $api_base \
    > $log_file 2>&1 &
echo "log file: $log_file"
