base_path="archrag" # input dataset path 
relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"
text_unit_filename="create_final_text_units.parquet"
output_dir="archrag_index" 
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=5
max_cluster_size=15
entity_second_embedding=True
api_key="0f4fb9015a8c458e8bfe9db1f70cdcd4.wH0YV4MtFKIpJLBe" #TODOsk-GPKz0LPEDWO6sJMT0
api_base="https://open.bigmodel.cn/api/paas/v4" #TODOhttps://api.mineguai.com/v1
engine="glm-5.1" # llm enginegpt-5.5

# If you have embedding model settings, uncomment and set them here
# And make sure to uncomment the corresponding line in the nohup command below
# --embedding_model $embedding_model --embedding_api_key $embedding_api_key --embedding_api_base $embedding_api_base \    
embedding_model="embedding-3" # TODO
embedding_api_key="cdfe55209a7f40819a7a8b9f2ece7910.vGQyG5kIhNXVAxJJ"
embedding_api_base="https://open.bigmodel.cn/api/paas/v4" # TODO

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
