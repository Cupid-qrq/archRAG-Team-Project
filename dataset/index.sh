base_path="archrag" # input dataset path 
relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"
output_dir="archrag_index" 
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=10
max_cluster_size=15
entity_second_embedding=True
api_key="sk-GPKz0LPEDWO6sJMT0" #TODO
api_base="http://localhost:8327/v1" #TODO
engine="gpt-5.5" # llm engine

# If you have embedding model settings, uncomment and set them here
# And make sure to uncomment the corresponding line in the nohup command below
# --embedding_model $embedding_model --embedding_api_key $embedding_api_key --embedding_api_base $embedding_api_base \    
embedding_model="BAAI/bge-m3" # TODO
embedding_api_key="dummy"
embedding_api_base="http://localhost:8080/v1" # TODO

augment_graph=True
cluster_method="weighted_leiden"
num_workers=10

log_file="./index.log"
python_file="src/index.py"

export CUDA_VISIBLE_DEVICES=7

nohup python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
    --entity_filename $entity_filename --output_dir $output_dir --wx_weight $wx_weight --m_du_scale $m_du_scale --max_level $max_level \
    --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
    --entity_second_embedding $entity_second_embedding \
    --engine $engine --num_workers $num_workers \
    --augment_graph $augment_graph --cluster_method $cluster_method \
    --embedding_model $embedding_model --embedding_api_key $embedding_api_key --embedding_api_base $embedding_api_base \
    --api_key $api_key --api_base $api_base \
    > $log_file 2>&1 &
echo "log file: $log_file"
