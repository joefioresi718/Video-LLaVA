import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# Surpress key warnings.
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")


def init_model():
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    cache_dir = 'cache_dir'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    return tokenizer, model, video_processor, context_len


def roll_padding_to_front(padded_input_ids, padding_value=0):
    padding_lengths = (padded_input_ids == padding_value).long().sum(dim=1)
    rolled_input_ids = torch.stack([torch.roll(input_id, shifts=padding_length.item()) for input_id, padding_length in zip(padded_input_ids, padding_lengths)])
    return rolled_input_ids


def run_videollava(model, tokenizer, video_processor, video, inp, conv=None):
    if conv is None:
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    roles = conv.roles

    # Load in video.
    video_tensor = video_processor([video, video], return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)


    print(f"{roles[1]}: {inp}")
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
        

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    return outputs, conv


def run_videollava_batch(model, tokenizer, video_processor, videos, inps, conv=None):
    inputs_ids = []
    stopping_criterias = []
    # Load in video.
    videos_tensor = video_processor(videos, return_tensors='pt')['pixel_values']
    videos_tensor = videos_tensor.to(model.device, dtype=torch.float16)

    for inp in inps:
        if conv is None:
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        roles = conv.roles

        print(f"{roles[1]}: {inp}")
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids.unsqueeze(0))
        inputs_ids.append(input_ids)
        stopping_criterias.append(stopping_criteria)

    padded_input_ids = pad_sequence(
        inputs_ids, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    ).cuda()
    # move padding tokens ahead
    rolled_input_ids = roll_padding_to_front(padded_input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            rolled_input_ids,
            images=videos_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=stopping_criterias)
        
    outputs = tokenizer.batch_decode(
        output_ids, 
        skip_special_tokens=True
    )
    outputs = [x.strip() for x in outputs]

    return outputs


def main():
    disable_torch_init()
    tokenizer, model, video_processor, context_len = init_model()

    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/Kayaking/v_Kayaking_g02_c02.avi'
    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/Kayaking/v_Kayaking_g04_c06.avi'
    # inp = 'Given the following question with answer choices, please select the most appropriate answer. \n'
    # inp += 'Question: Is the baby wearing a shirt?\n'
    # inp += 'Answer Choices: (a) Yes, (b) No'

    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/BabyCrawling/v_BabyCrawling_g05_c05.avi'
    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/BabyCrawling/v_BabyCrawling_g07_c05.avi'
    # inp = 'Given the following question with answer choices, please select the most appropriate answer. \n'
    # inp += 'Question: Is the baby wearing a shirt?\n'
    # inp += 'Answer Choices: (a) Yes, (b) No'
    videos = []
    inps = []

    video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/ParallelBars/v_ParallelBars_g04_c01.avi'
    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/BalanceBeam/v_BalanceBeam_g06_c07.avi'
    inp = 'Given the following question with answer choices, please select the most appropriate answer. \n'
    inp += 'Question: Is the main subject using a balance beam?\n'
    inp += 'Answer Choices: (a) Yes, (b) No'

    videos.append(video)
    inps.append(inp)

    # video = '/home/jo869742/PythonProjects/datasets/hmdb51/videos/catch/Goalkeeper_Training_Day_#_7_catch_f_cm_np1_ba_bad_2.avi'
    # video = '/home/jo869742/PythonProjects/datasets/hmdb51/videos/catch/Goalkeeper_Training_Day_#_2_catch_f_cm_np1_ba_bad_4.avi'
    # inp = 'Is the camera placed directly behind the goal? '
    # inp += '(a) Yes (b) No'
    inp = 'Describe a scene where someone is brushing their hair.'
    video = '/home/jo869742/PythonProjects/datasets/hmdb51/videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_1.avi'

    videos.append(video)
    inps.append(inp)

    outputs, conv = run_videollava_batch(model, tokenizer, video_processor, videos, inps)
    print(outputs)
    exit()

    conv.messages[-1][-1] = outputs
    inp = 'Why did you choose that answer?'
    
    outputs, conv = run_videollava(model, tokenizer, video_processor, video, inp, conv)
    print(outputs[:-4])

    exit()

    conv.messages[-1][-1] = outputs
    inp = 'The correct answer is actually BlowDryHair. Can you explain what biases you may have had when making your class choice?'
    # inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp

    outputs, conv = run_videollava(model, tokenizer, video_processor, video, inp, conv)
    print(outputs[:-4])


if __name__ == '__main__':
    main()