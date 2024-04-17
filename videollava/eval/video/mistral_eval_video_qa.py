import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-mistral")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--model_name", default="mistralai/Mixtral-8x7B-Instruct-v0.1", type=str, help="HuggingFace model name.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir, model, tokenizer, args):
    """
    Evaluates question and answer pairs using Mistral
    Returns a score for correctness.
    """
    for file in caption_files:
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']

        chat = [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            }
        ]

        prompt = f'<s>[INST] {chat[0]["content"].strip()}\n\n{chat[1]["content"].strip()} [/INST]'
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, use_cache=True, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
        # prompt = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
        # outputs = model.generate(prompt, use_cache=True, max_new_tokens=150)
        response_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_message = response_message.split('[/INST]')[-1].split(tokenizer.eos_token)[0].strip().split('\n')[0].strip()
        # print(response_message)
        try:
            # Convert response to a Python dictionary.
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    '''
    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    llm_model = args.model_name

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(llm_model, device_map='cuda', torch_dtype=torch.bfloat16, cache_dir='cache_dir', quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(llm_model, cache_dir='cache_dir')
    if 'mistralai' in llm_model:
        tokenizer.pad_token = tokenizer.eos_token

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break

            annotate(prediction_set, incomplete_files, output_dir, model, tokenizer, args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()

