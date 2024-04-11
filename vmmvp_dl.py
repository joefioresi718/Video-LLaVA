import glob
import json
import numpy as np
import os, sys, traceback
import pandas as pd
import random
import time
import torch
import decord
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans


decord.bridge.set_bridge('torch')

class vmmvp_dataset(Dataset):
    def __init__(self, dataset='all', dataset_folder='V-MMVP', vid_file='V-MMVP_final.csv', num_frames=8):
        self.data = pd.read_csv(os.path.join(dataset_folder, vid_file))
        self.dataset = dataset
        self.dataset_folder = dataset_folder
        self.num_frames = num_frames

        if self.dataset != 'all':
            self.data = self.all_data[self.all_data['dataset'] == self.dataset]
        
        self.transform = trans.Compose([
                trans.Lambda(lambda x: x / 255.0),
                trans.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                trans.Resize(size=224, antialias=False),
                trans.CenterCrop(size=224),
                # trans.RandomHorizontalFlip(p=0.5)
            ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid1_name = row['video1']
        vid2_name = row['video2']
        vid1_class = row['video1_class']
        vid2_class = row['video2_class']
        clip_similarity = row['clip_similarity']
        vssl_similarity = row['vssl_similarity']
        pair_path = row['pair_path']
        vid1_path = os.path.join(self.dataset_folder, pair_path, vid1_name)
        vid2_path = os.path.join(self.dataset_folder, pair_path, vid2_name)

        vid1, vid2, frame_list1, frame_list2 = self.process_data(vid1_path, vid2_path)
        vids = torch.stack([vid1, vid2])
        frame_lists = torch.stack([frame_list1, frame_list2])
        vid_classes = [vid1_class, vid2_class]

        return vids, frame_lists, vid_classes, clip_similarity, vssl_similarity


    def process_data(self, vid1_path, vid2_path):
        vr1 = decord.VideoReader(vid1_path)
        vr2 = decord.VideoReader(vid2_path)

        frame_count1 = len(vr1)
        frame_count2 = len(vr2)
        
        frame_id_list1 = np.linspace(0, frame_count1-1, self.num_frames, dtype=int)
        frame_id_list2 = np.linspace(0, frame_count2-1, self.num_frames, dtype=int)

        vid1 = vr1.get_batch(frame_id_list1) # (T, H, W, C)
        vid2 = vr2.get_batch(frame_id_list2) # (T, H, W, C)

        vid1 = vid1.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        vid2 = vid2.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        vid1 = self.transform(vid1)
        vid2 = self.transform(vid2)

        vid1 = vid1.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        vid2 = vid2.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

        return vid1, vid2, torch.from_numpy(frame_id_list1), torch.from_numpy(frame_id_list2)
    

def collate_fn(batch):
    vids, frame_lists, vid_classes, clip_similarities, vssl_similarities = zip(*batch)
    vids = torch.stack(vids)
    frame_lists = torch.stack(frame_lists)
    clip_similarities = torch.tensor(clip_similarities)
    vssl_similarities = torch.tensor(vssl_similarities)

    return vids, frame_lists, vid_classes, clip_similarities, vssl_similarities
    

if __name__ == '__main__':
    dataset = vmmvp_dataset()
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for vids, frame_lists, vid_classes, clip_similarities, vssl_similarities in dataloader:
        print(vids.shape, frame_lists.shape, vid_classes, clip_similarities, vssl_similarities)
        break
