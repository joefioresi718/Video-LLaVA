import clip
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
    video = '/home/jo869742/PythonProjects/datasets/UCF101/Videos/BabyCrawling/v_BabyCrawling_g02_c03.avi'
    inp = 'Describe the activity in the video.'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # print(model)
    # exit()

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
                texts = tokenizer(texts, return_tensors='pt').cuda() #tokenize
                # class_embeddings = model.model(texts.input_ids.cuda())[0] #embed with text encoder
                # texts = [tokenizer_X_token(DEFAULT_X_TOKEN['VIDEO'] + '\n' + template.format(classname), tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0) for template in templates][0]
                class_embeddings = model.model(input_ids=texts.input_ids, attention_mask=texts.attention_mask, past_key_values=None, inputs_embeds=None)[0]
                print(class_embeddings.shape)
                exit()
                # class_embeddings = model.model.layers(class_embeddings)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings /= class_embeddings.norm()
                # class_embeddings = model.model.norm(class_embeddings)
                # class_embeddings = model.norm(model.model(texts.input_ids.cuda(), texts.attention_mask)[0]) #embed with text encoder
                # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                # class_embedding = class_embeddings.mean(dim=0)
                # class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    zeroshot_weights = zeroshot_classifier(ucf101_classes, templates)

    # Load in video.
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

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
        vid_embed = model.model.video_tower(tensor)
        mm_embed = model.model.mm_projector(vid_embed)[0]
        mm_embed = mm_embed.mean(dim=0)
        mm_embed /= mm_embed.norm()

        # output_ids = model.generate(
        #     input_ids,
        #     images=[tensor, key],
        #     do_sample=True,
        #     temperature=0.1,
        #     max_new_tokens=1024,
        #     use_cache=True,
        #     stopping_criteria=[stopping_criteria])
        
    logits = 100. * mm_embed.unsqueeze(0) @ zeroshot_weights
    topk = (1,)
    pred = logits.topk(max(topk), 1, True, True)[1].t()
    print('Prediction', pred[0][0], ucf101_classes[pred[0][0]])
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    # print(outputs[:-4])

    # texts = clip.tokenize([outputs[:-4]], truncate=True).cuda() #tokenize
    # text_features = clip_model.encode_text(texts)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # logits = 100. * text_features @ zeroshot_weights
    # topk = (1,)

    # pred = logits.topk(max(topk), 1, True, True)[1].t()
    # print('Prediction', pred[0][0], ucf101_classes[pred[0][0]])

if __name__ == '__main__':
    main()