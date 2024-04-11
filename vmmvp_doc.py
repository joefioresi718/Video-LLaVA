import pandas as pd
import os.path


ucf_pairs = pd.read_csv('csvs/CLIPblind_pairs_ucf_vitl_final.csv')
kin_pairs = pd.read_csv('csvs/CLIPblind_pairs_k400_vitl_final.csv')

print(ucf_pairs.head())
print(kin_pairs.head())

k400_file = open('/home/jo869742/PythonProjects/datasets/kinetics-dataset/annotation_test_fullpath_resizedvids.txt', 'r').read().splitlines()
# Class names can be multiple words, so we need to split the line by spaces and take the last few elements.
k400_classes = {os.path.basename(line.split(' ')[0]): ' '.join(line.split(' ')[2:]) for line in k400_file}

ucf_pairs['dataset'] = 'ucf101'
kin_pairs['dataset'] = 'kinetics400'

# Add pair paths.
ucf_pair_paths = [f'ucf101_pairs/ucf{idx:03d}' for idx in range(len(ucf_pairs))]
kin_pair_paths = [f'k400_pairs/kin{idx:03d}' for idx in range(len(kin_pairs))]
ucf_pairs['pair_path'] = ucf_pair_paths
kin_pairs['pair_path'] = kin_pair_paths

# Remove preference score column.
ucf_pairs.drop(columns='preference_score', inplace=True)
kin_pairs.drop(columns='preference_score', inplace=True)

# Add class names for each video.
ucf_pairs['video1_class'] = ucf_pairs['video1'].str.split('_').str[1].str.replace('HandStand', 'Handstand')
ucf_pairs['video2_class'] = ucf_pairs['video2'].str.split('_').str[1].str.replace('HandStand', 'Handstand')
kin_pairs['video1_class'] = [k400_classes[video] for video in kin_pairs['video1']]
kin_pairs['video2_class'] = [k400_classes[video] for video in kin_pairs['video2']]
# kin_pairs['video1_class'] = kin_pairs['video1'].str.split('_').str[1]
# kin_pairs['video2_class'] = kin_pairs['video2'].str.split('_').str[1]

# Rearrange columns.
ucf_pairs = ucf_pairs[['dataset', 'pair_path', 'video1', 'video1_class', 'video2', 'video2_class', 'clip_similarity', 'vssl_similarity']]

# Concatenate the two DataFrames.
all_pairs = pd.concat([ucf_pairs, kin_pairs], ignore_index=True)

# Save the refined DataFrame to a new CSV file.
all_pairs.to_csv('V-MMVP_final.csv', index=False)