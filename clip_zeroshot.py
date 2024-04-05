import numpy as np
import os.path
import torch
import clip
import glob
from tqdm import tqdm
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import warnings
warnings.filterwarnings("ignore")

import config as cfg
from llava_dl import *
import params.params_baseline as params


# Load CLIP model.
clip_model, preprocess = clip.load('ViT-L/14')

# Load LLAVA model.
disable_torch_init()
llava_input = 'Describe the activity in the video.'
model_path = 'LanguageBind/Video-LLaVA-7B'
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device='cuda')
video_processor = processor['video']
conv_mode = "llava_v1"
llava_input = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + llava_input
conv = conv_templates[conv_mode].copy()
roles = conv.roles

conv.append_message(conv.roles[0], llava_input)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
key = ['video']

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
    'a human action of {}',
]

# Create dataloader.
ds = baseline_val_dataloader(params=params, dataset='ucf101', shuffle=False, data_percentage=1.0, processor=video_processor)
dl = DataLoader(ds, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_val, num_workers=params.num_workers)

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

zeroshot_weights = zeroshot_classifier(ucf101_classes, templates)

with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (videos, target) in enumerate(tqdm(dl)):
        videos = videos.to('cuda', dtype=torch.float16)
        # videos = videos.permute(0, 2, 1, 3, 4)
        target = target.cuda()
        outputs = []
        for video in videos:
            output_ids = model.generate(
                input_ids,
                images=videos,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

            outputs.append(tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()[:-4])
            print(outputs[-1])
        # exit()
        # predict
        texts = clip.tokenize(outputs, truncate=True).cuda() #tokenize
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = 100. * text_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += videos.size(0)

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")
