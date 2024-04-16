# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import argparse
import json
import re
import time

from huggingface_hub import login
login('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')

# llm_model = 'meta-llama/Llama-2-7b-chat-hf'
# llm_model = 'google/gemma-1.1-7b-it'
# llm_model = 'mistralai/Mistral-7B-Instruct-v0.2'
llm_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(llm_model)
if 'mistralai' in llm_model:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(llm_model, device_map='cuda', torch_dtype=torch.bfloat16)


# Create the parser
parser = argparse.ArgumentParser(description='Process pandas file path.')

# Add arguments
parser.add_argument('--answer_file', default = 'V-MMVP_ft/V-MMVP_ft_results_videochatgpt.csv', help='Path to the pandas file')

# Parse arguments
args = parser.parse_args()

# Define a function to query the OpenAI API and evaluate the answer
def get_yes_no_answer(question):
    system_prompt = 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the one word answer. For example, your response should look like this: "no".'
    additional_context = ''
    input_text = f'{system_prompt} {question} {additional_context}'
    question = f'{question} {additional_context}'
    if llm_model.startswith('meta-llama') or llm_model.startswith('mistralai'):
        chat = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ]
    else:
        chat = [
            {
                'role': 'user',
                'content': input_text
            }
        ]

    if llm_model.startswith('mistralai'):
        prompt = f'<s>[INST] {chat[0]["content"].strip()}\n\n{chat[1]["content"].strip()} [/INST]'
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        # prompt = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, use_cache=True, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    else:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

    if llm_model.startswith('meta-llama') or llm_model.startswith('mistralai'):
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split('[/INST]')[-1].split(tokenizer.eos_token)[0].replace('.', '').strip().split(' ')[0].replace('(', ' ').replace(')', ' ').strip()
    elif llm_model.startswith('google'):
        answer = tokenizer.decode(outputs[0])
        answer = answer.split('<start_of_turn>model')[-1].split('<eos>')[0].strip()
    # elif llm_model.startswith('mistralai'):
    #     print(answer)
    # print(answer)

    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."


num_correct, num_total = 0, 0

file = pd.read_csv(args.answer_file)

for i, row in file.iterrows():
    dataset, pair_path, video1, video2, question, options, answer1, answer2, v1_correct, v2_correct, clip_similarity, vssl_similarity = row

    v1_grader_question = f'Given the following question: {question} {options}, the correct answer is {v1_correct}. Does the following answer correctly answer the question, answer: {answer1}?'
    v1_gpt_grade = get_yes_no_answer(v1_grader_question)

    v2_grader_question = f'Given the following question: {question} {options}, the correct answer is {v2_correct}. Does the following answer correctly answer the question, answer: {answer2}?'
    v2_gpt_grade = get_yes_no_answer(v2_grader_question)

    num_total += 1
    if v1_gpt_grade == 'yes' and v2_gpt_grade == 'yes':
        print(pair_path, '-- both correct!')
        num_correct += 1
    else:
        print(pair_path, '-- incorrect.')

print(f'The accuracy is {num_correct} of {num_total} : {(num_correct/num_total)*100:.2f}%')