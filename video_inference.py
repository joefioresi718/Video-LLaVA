import clip
import pandas as pd
import torch
from tqdm import tqdm
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import warnings
warnings.filterwarnings("ignore")

import config as cfg


def main():
    disable_torch_init()
    # video = 'llava/serve/examples/sample_demo_1.mp4'
    # video = cfg.ucf101_path + '/Videos/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    # video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/BlowDryHair/v_BlowDryHair_g02_c01.avi'
    video = '/home/jo869742/PythonProjects/datasets/BiasEvaluation/ARAS/videos/golf_driving/WEHjZfTwBa8_1.mp4'
    inp = 'Describe the activity in the video.'
    # inp = 'Can you describe a made up scene where a person is brushing their hair?'
    kclasses = pd.read_csv('k400_classes.csv')
    kclasses = kclasses['name'].tolist()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    cache_dir = 'cache_dir'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Load CLIP model, set params.
    clip_model, _ = clip.load('ViT-L/14')

    ucf101_classes = [
        'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 
        'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 
        'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 
        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 
        'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 
        'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 
        'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 
        'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 
        'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 
        'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 
        'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 
        'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 
        'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 
        'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 
        'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 
        'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 
        'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 
        'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 
        'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 
        'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 
        'YoYo'
    ]

    templates = [
        lambda c: f'a bad video of a {c}.',
        lambda c: f'a video of many {c}.',
        lambda c: f'a sculpture of a {c}.',
        lambda c: f'a video of the hard to see {c}.',
        lambda c: f'a low resolution video of the {c}.',
        lambda c: f'a rendering of a {c}.',
        lambda c: f'graffiti of a {c}.',
        lambda c: f'a bad video of the {c}.',
        lambda c: f'a cropped video of the {c}.',
        lambda c: f'a tattoo of a {c}.',
        lambda c: f'the embroidered {c}.',
        lambda c: f'a video of a hard to see {c}.',
        lambda c: f'a bright video of a {c}.',
        lambda c: f'a video of a clean {c}.',
        lambda c: f'a video of a dirty {c}.',
        lambda c: f'a dark video of the {c}.',
        lambda c: f'a drawing of a {c}.',
        lambda c: f'a video of my {c}.',
        lambda c: f'the plastic {c}.',
        lambda c: f'a video of the cool {c}.',
        lambda c: f'a close-up video of a {c}.',
        lambda c: f'a black and white video of the {c}.',
        lambda c: f'a painting of the {c}.',
        lambda c: f'a painting of a {c}.',
        lambda c: f'a pixelated video of the {c}.',
        lambda c: f'a sculpture of the {c}.',
        lambda c: f'a bright video of the {c}.',
        lambda c: f'a cropped video of a {c}.',
        lambda c: f'a plastic {c}.',
        lambda c: f'a video of the dirty {c}.',
        lambda c: f'a jpeg corrupted video of a {c}.',
        lambda c: f'a blurry video of the {c}.',
        lambda c: f'a video of the {c}.',
        lambda c: f'a good video of the {c}.',
        lambda c: f'a rendering of the {c}.',
        lambda c: f'a {c} in a video game.',
        lambda c: f'a video of one {c}.',
        lambda c: f'a doodle of a {c}.',
        lambda c: f'a close-up video of the {c}.',
        lambda c: f'a video of a {c}.',
        lambda c: f'the origami {c}.',
        lambda c: f'the {c} in a video game.',
        lambda c: f'a sketch of a {c}.',
        lambda c: f'a doodle of the {c}.',
        lambda c: f'a origami {c}.',
        lambda c: f'a low resolution video of a {c}.',
        lambda c: f'the toy {c}.',
        lambda c: f'a rendition of the {c}.',
        lambda c: f'a video of the clean {c}.',
        lambda c: f'a video of a large {c}.',
        lambda c: f'a rendition of a {c}.',
        lambda c: f'a video of a nice {c}.',
        lambda c: f'a video of a weird {c}.',
        lambda c: f'a blurry video of a {c}.',
        lambda c: f'a cartoon {c}.',
        lambda c: f'art of a {c}.',
        lambda c: f'a sketch of the {c}.',
        lambda c: f'a embroidered {c}.',
        lambda c: f'a pixelated video of a {c}.',
        lambda c: f'itap of the {c}.',
        lambda c: f'a jpeg corrupted video of the {c}.',
        lambda c: f'a good video of a {c}.',
        lambda c: f'a plushie {c}.',
        lambda c: f'a video of the nice {c}.',
        lambda c: f'a video of the small {c}.',
        lambda c: f'a video of the weird {c}.',
        lambda c: f'the cartoon {c}.',
        lambda c: f'art of the {c}.',
        lambda c: f'a drawing of the {c}.',
        lambda c: f'a video of the large {c}.',
        lambda c: f'a black and white video of a {c}.',
        lambda c: f'the plushie {c}.',
        lambda c: f'a dark video of a {c}.',
        lambda c: f'itap of a {c}.',
        lambda c: f'graffiti of the {c}.',
        lambda c: f'a toy {c}.',
        lambda c: f'itap of my {c}.',
        lambda c: f'a video of a cool {c}.',
        lambda c: f'a video of a small {c}.',
        lambda c: f'a tattoo of the {c}.',
    ]

    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = clip_model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
                
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    # zeroshot_weights = zeroshot_classifier(ucf101_classes, templates)

    # Load in video.
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    
    # inp = 'Please describe the activity in this video.'
    inp = 'Given the following action class choices, please select the most appropriate class for the video. The class choices are separated by comma and many of the classes are made up of multiple words. Return only the selected class choice, nothing else. \n' + ', '.join(kclasses)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
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
    print(outputs[:-4])

    conv.messages[-1][-1] = outputs
    # inp = 'Explain your reasoning for making this class choice.'
    # inp = 'Given the following class choices, please select the most appropriate class for the video: \n' + ', '.join(kclasses)
    # inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    inp = 'Ignore the video content. Can you tell me what the first class choice presented to you was?'
    
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
    print(outputs[:-4])

    exit()

    conv.messages[-1][-1] = outputs
    inp = 'The correct answer is actually BlowDryHair. Can you explain what biases you may have had when making your class choice?'
    # inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    
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
    print(outputs[:-4])

    # texts = clip.tokenize([outputs[:-4]], truncate=True).cuda() #tokenize
    # text_features = clip_model.encode_text(texts)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # logits = 100. * text_features @ zeroshot_weights
    # topk = (1,)

    # pred = logits.topk(max(topk), 1, True, True)[1].t()
    # print('Prediction', pred[0][0], ucf101_classes[pred[0][0]])

if __name__ == '__main__':
    main()