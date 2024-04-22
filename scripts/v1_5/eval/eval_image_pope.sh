#!/bin/bash


CKPT_NAME="videollava-7b-lora"
CKPT="checkpoints/${CKPT_NAME}"
# CKPT="LanguageBind/Video-LLaVA-7B"
EVAL="eval"
EVAL_COCO="/datasets/MSCOCO"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
python3 -m videollava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL_COCO}/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-base ${MODEL_BASE} \

python3 videollava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl
