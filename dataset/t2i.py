import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image 


class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)
        
        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)            
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name 


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl file path
        for lst_name in sorted(os.listdir(args.data_path)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(args.data_path, lst_name)
            valid_file_path.append(file_path)           
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
        if self.short_t5_feat_path is not None:
            self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
        else:
            self.short_t5_feat_path_base = self.t5_feat_path_base
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")                
        except:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if min(img.size) < self.image_size:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            return img, t5_feat_padding, attn_mask, torch.tensor(valid)

        if self.transform is not None:
            img = self.transform(img)
        
        t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        if torch.rand(1) < 0.3:
            t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)
        
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        if os.path.isfile(t5_file):
            try:
                t5_feat = torch.from_numpy(np.load(t5_file))
                t5_feat_len = t5_feat.shape[1] 
                feat_len = min(self.t5_feature_max_len, t5_feat_len)
                t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
                emb_mask = torch.zeros((self.t5_feature_max_len,))
                emb_mask[-feat_len:] = 1
                attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
                T = self.t5_feature_max_len
                attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
                eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
                attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
                attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
                valid = 1
            except:
                img, t5_feat_padding, attn_mask, valid = self.dummy_data()
        else:
            img, t5_feat_padding, attn_mask, valid = self.dummy_data()
            
        return img, t5_feat_padding, attn_mask, torch.tensor(valid)


class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass




def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)

def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)

def build_t2i_code(args):
    return Text2ImgDatasetCode(args)