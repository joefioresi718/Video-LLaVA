import numpy as np
import os.path
import pandas as pd
import torch
# Surpress key warnings.
from transformers import logging
logging.set_verbosity_error()

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vmmvp_dl import vmmvp_dataset, collate_fn


def init_model(model_path='LanguageBind/Video-LLaVA-7B', device='cpu'):
    cache_dir = 'cache_dir'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    return tokenizer, model, video_processor, context_len


def main(model_path):
    disable_torch_init()
    device = 'cuda:0'
    tokenizer, model, video_processor, context_len = init_model(model_path=model_path, device=device)

    # dataset = vmmvp_dataset()
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    dataset_path = 'V-MMVP_ft'
    dataset = pd.read_csv('V-MMVP_ft/V-MMVP_ft_final.csv')

    all_results = []
    for i, row in dataset.iterrows():
        if row['dataset'] == 'kinetics400':
            break
        videos = video_processor([os.path.join(dataset_path, row['pair_path'], row['video1']), os.path.join(dataset_path, row['pair_path'], row['video2'])], return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
        video1, video2 = videos
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)

        inp = row['question'] + ' ' + row['options']

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        roles = conv.roles

        print(f"{roles[1]}: {inp}")
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_id)

        with torch.inference_mode():
            output_ids1 = model.generate(
                input_id,
                images=video1,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
            output_ids2 = model.generate(
                input_id,
                images=video2,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
        outputs1 = tokenizer.decode(output_ids1[0, input_id.shape[1]:]).strip()
        outputs2 = tokenizer.decode(output_ids2[0, input_id.shape[1]:]).strip()
        if outputs1.endswith(stop_str):
            outputs1 = outputs1[:-len(stop_str)]
        if outputs2.endswith(stop_str):
            outputs2 = outputs2[:-len(stop_str)]
        outputs1 = outputs1.strip()
        outputs2 = outputs2.strip()

        print(outputs1)
        print(outputs2)

        result = {
            'dataset': row['dataset'],
            'pair_path': row['pair_path'],
            'video1': row['video1'],
            'video2': row['video2'],
            'question': row['question'],
            'options': row['options'],
            'answer1': outputs1,
            'answer2': outputs2,
            'v1_correct': row['v1_correct_answer'],
            'v2_correct': row['v2_correct_answer'],
            'clip_similarity': row['clip_similarity'],
            'vssl_similarity': row['vssl_similarity']
        }

        print(result)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('V-MMVP_ft/V-MMVP_ft_results.csv', index=False)


if __name__ == '__main__':
    model_path = 'LanguageBind/Video-LLaVA-7B'
    main(model_path)
