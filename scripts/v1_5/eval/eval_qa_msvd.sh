

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name="videollava-7b"
pred_path="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/merge.jsonl"
# output_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/mixtral_8x7B"
output_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/llama3_70B"
output_json="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/${output_name}/results.json"
# model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name="mistralai/Mistral-7B-Instruct-v0.2"
model_name="meta-llama/Meta-Llama-3-70B-Instruct"



python3 videollava/eval/video/llama_eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --model_name ${model_name}
