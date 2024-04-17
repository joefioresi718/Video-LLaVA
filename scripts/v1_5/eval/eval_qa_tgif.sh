

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name="Video-LLaVA-7B"
pred_path="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/mixtral_8x7B"
output_json="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/results.json"
model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name="mistralai/Mistral-7B-Instruct-v0.2"



python3 videollava/eval/video/mistral_eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --model_name ${model_name}
