# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import warnings
import pickle
import torch
#import smplx
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFile
import random
from utils.constants import AGORA_DIR, AGORA_IMG_SIZE
from utils.image import normalize_rgb, denormalize_rgb
ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict

class AGORA(Dataset):
    def __init__(self,
                 split='training',
                 training=True,
                 img_size=512,
                 root_dir=AGORA_DIR,
                 subsample=1,
                 crops=[0],
                 flip=1,
                 n=-1,
                 *args, **kwargs
                 ):
        super().__init__()
        
        self.name = 'agora'
        self.training = training
        self.img_size = img_size
        self.subsample = subsample
        self.crops = crops  # 0 is the default crop (no crop)
        self.flip = flip    # 1 means flipping is enabled
        
        assert split in ['training', 'validation', 'test']
        self.split = split
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, f"{self.split}")
        
        # Build a dictionary mapping image file names to their full paths.
        self.image_path_dict = self.build_image_path_dict()
        
        # Load NPZ annotations lazily via memory mapping.
        npz_path = "data/agora-bfh.npz"
        self.npz_data = np.load(npz_path, mmap_mode='r')
        
        # Build an index: map each image name to the row indices (each row corresponds to one person).
        self.index_dict = defaultdict(list)
        for idx, imgname in enumerate(self.npz_data['imgname']):
            # Optionally, ensure the image exists on disk.
            if imgname in self.image_path_dict:
                self.index_dict[imgname].append(idx)
        
        # Use the keys from the index as the list of images.
        self.imagenames = sorted(list(self.index_dict.keys()))
        print(f"Number of images available: {len(self.imagenames)}")

        #Reset the reader for later multi-worker dataloading
        self.npz_data = None
        
        if n >= 0:
            self.imagenames = self.imagenames[:n]
        
        if self.subsample > 1:
            self.imagenames = [self.imagenames[k] for k in np.arange(0, len(self.imagenames), self.subsample).tolist()]

    def build_image_path_dict(self):
        image_path_dict = {}
        for dirpath, dirnames, filenames in os.walk(self.image_dir):
            for fn in filenames:
                image_path_dict[fn] = os.path.join(dirpath, fn)
        return image_path_dict

    def __len__(self):
        return len(self.imagenames)
        
    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.imagenames)}"

    def __getitem__(self, idx):
        # Lazy load the NPZ data, with a different file handle per worker.
        if not hasattr(self, 'npz_data') or self.npz_data is None:
            npz_path = "data/agora-bfh.npz"
            self.npz_data = np.load(npz_path, mmap_mode='r')

        if self.training:
            idx = random.choice(range(len(self.imagenames)))
        imagename = self.imagenames[idx]
        
        # Get the list of indices (rows) for this image from the NPZ file.
        indices = self.index_dict[imagename]
        
        # Retrieve image path from the prebuilt dictionary.
        img_path = self.image_path_dict[imagename]
        
        # Use the first row to get camera intrinsics.
        first_idx = indices[0]
        K = self.npz_data['cam_int'][first_idx]
        focal = np.asarray([K[0, 0], K[1, 1]])
        princpt = np.asarray([K[0, -1], K[1, -1]])
        
        # Build the image-level annotation.
        annot = {
            'imgname': imagename,
            'imgpath': img_path,
            'focal': focal,
            'princpt': princpt,
            'size': AGORA_IMG_SIZE,
        }
        
        persons = []
        # Process each person (each row) for this image.
        for i in indices:
            pose = self.npz_data['pose_cam'][i]
            shape_array = self.npz_data['shape'][i]
            trans_cam = self.npz_data['trans_cam'][i]
            H_array = self.npz_data['cam_ext'][i]
            
            person = {
                'smplx_root_pose': pose[:3].reshape(1, 3),
                'smplx_body_pose': pose[3:66].reshape(21, 3),
                'smplx_jaw_pose': pose[66:69].reshape(1, 3),
                'smplx_leye_pose': pose[69:72].reshape(1, 3),
                'smplx_reye_pose': pose[72:75].reshape(1, 3),
                'smplx_left_hand_pose': pose[75:120].reshape(15, 3),
                'smplx_right_hand_pose': pose[120:165].reshape(15, 3),
                'smplx_shape': shape_array.reshape(11),
                'smplx_gender': 'neutral',
                'smplx_transl': (trans_cam + H_array[:, 3][:3]).reshape(3),
            }
            persons.append(person)
        annot['persons'] = persons

        real_width, real_height = annot['size']
        K_matrix = np.eye(3)
        princpt_norm = annot['princpt'].copy()
        princpt_width = princpt_norm[0] / real_width
        princpt_height = princpt_norm[1] / real_height
        K_matrix[[0, 1], [-1, -1]] = self.img_size * np.asarray([princpt_width, princpt_height])
        K_matrix[[0, 1], [0, 1]] = annot['focal'] / (max(real_width, real_height) / self.img_size)
        
        # Load and preprocess the image.
        img_pil = Image.open(img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
            
        # (Optional: BEDLAM-specific rotation fix)
        if self.name == 'bedlam' and 'closeup' in imagename and self.split != 'test':
            img_pil = img_pil.rotate(-90, expand=True)
        
        # Flipping augmentation.
        do_flip = self.flip and random.choice([0, 1]) and self.training
        if do_flip:
            img_pil = ImageOps.mirror(img_pil)
            K_matrix[0, -1] = self.img_size - K_matrix[0, -1]
        
        # Cropping (default: zero padding to make square).
        crop = random.choice(self.crops) if self.training else 0
        if crop == 0:
            img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size))
            img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size))
        else:
            raise NotImplementedError("Custom cropping not implemented.")
        
        img_array = np.asarray(img_pil)
        img_array = normalize_rgb(img_array, imagenet_normalization=True)
        
        # Update annotation with camera matrix.
        annot['K'] = K_matrix
        annot.pop('princpt', None)
        annot.pop('focal', None)
        
        # Process persons: sort by depth (z coordinate of translation).
        _humans = annot['persons'].copy()
        if self.training:
            humans = [hum for hum in _humans if hum['smplx_transl'][-1] > 0.01]
        else:
            humans = _humans.copy()
        l_dist = [hum['smplx_transl'][-1] for hum in humans]
        sorted_indices = [i for i, _ in sorted(enumerate(l_dist), key=lambda x: x[1])]
        annot['persons'] = [humans[i] for i in sorted_indices]
        
        # Update smplx_gender to numerical id (e.g., neutral=0).
        for hum in annot['persons']:
            hum['smplx_gender_id'] = np.asarray({'neutral': 0}[hum['smplx_gender']])
        
        # If flipping, update the human parameters accordingly.
        if do_flip:
            flipped_humans = []
            for hum in annot['persons']:
                _hum = hum.copy()
                for key in ['smplx_root_pose', 'smplx_body_pose', 'smplx_left_hand_pose',
                            'smplx_right_hand_pose', 'smplx_jaw_pose', 'smplx_transl',
                            'smplx_leye_pose', 'smplx_reye_pose']:
                    _hum.pop(key, None)
                _hum['smplx_transl'] = hum['smplx_transl'].copy()
                _hum['smplx_transl'][0] = -hum['smplx_transl'][0]
                
                _hum['smplx_root_pose'] = hum['smplx_root_pose'].copy()
                _hum['smplx_root_pose'][:, 1:3] *= -1
                
                _hum['smplx_jaw_pose'] = hum['smplx_jaw_pose'].copy()
                _hum['smplx_jaw_pose'][:, 1:3] *= -1
                
                _pose = hum['smplx_body_pose'].copy()
                orig_flip_pairs = ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20))
                for pair in orig_flip_pairs:
                    _pose[pair[0], :], _pose[pair[1], :] = _pose[pair[1], :].copy(), _pose[pair[0], :].copy()
                _pose[:, 1:3] *= -1
                _hum['smplx_body_pose'] = _pose
                
                lhand, rhand = hum['smplx_left_hand_pose'].copy(), hum['smplx_right_hand_pose'].copy()
                lhand[:, 1:3] *= -1
                rhand[:, 1:3] *= -1
                _hum['smplx_left_hand_pose'] = lhand
                _hum['smplx_right_hand_pose'] = rhand
                
                leye, reye = hum['smplx_leye_pose'].copy(), hum['smplx_reye_pose'].copy()
                leye[:, 1:3] *= -1
                reye[:, 1:3] *= -1
                _hum['smplx_leye_pose'] = leye
                _hum['smplx_reye_pose'] = reye
                
                flipped_humans.append(_hum)
            annot['persons'] = flipped_humans
        
        return img_array, annot
    

def collate_fn(batch, *args, **kwargs):
    # Number of samples in the batch
    bs = len(batch)
    y = {}

    # Collect images and image-level information
    img_array = torch.as_tensor(np.stack([sample[0] for sample in batch])).float()
    y['imagename'] = np.stack([sample[1]['imgname'] for sample in batch])
    y['K'] = torch.as_tensor(np.stack([sample[1]['K'] for sample in batch])).float()

    # Number of persons per image
    n_persons = [len(sample[1]['persons']) for sample in batch]
    y['n_persons'] = torch.as_tensor(n_persons).float()
    max_persons = int(max(n_persons))

    # Build validity mask for persons (1 for valid, 0 for padded entries)
    valid_persons = [
        np.concatenate([np.ones(n, dtype=np.float32), np.zeros(max_persons - n, dtype=np.float32)])
        for n in n_persons
    ]
    y['valid_humans'] = torch.as_tensor(np.stack(valid_persons)).float()

    # Determine all keys from the persons dictionaries and record expected shapes.
    all_keys = set()
    key2shape = {}
    for sample in batch:
        for person in sample[1]['persons']:
            for k, v in person.items():
                if isinstance(v, np.ndarray):
                    all_keys.add(k)
                    key2shape[k] = v.shape

    all_keys = list(all_keys)

    # For each person-key, stack the data from each sample with zero-padding.
    for k in all_keys:
        batch_values = []
        for sample in batch:
            persons = sample[1]['persons']
            if len(persons) == 0:
                # No persons for this image; create a zero array with zero rows.
                value = np.zeros((0,) + key2shape[k], dtype=np.float32)
            else:
                value = np.stack([person[k] for person in persons])
            # Zero-pad along the first dimension (number of persons)
            pad_size = max_persons - value.shape[0]
            if pad_size > 0:
                pad_shape = (pad_size,) + value.shape[1:]
                pad = np.zeros(pad_shape, dtype=np.float32)
                value = np.concatenate([value, pad], axis=0)
            batch_values.append(value)
        # Stack batch values and convert to tensor.
        y[k] = torch.as_tensor(np.stack(batch_values)).float()

    return img_array, y

'''
def visualize(split='validation', i=1500, res=None, extension='png', training=0, img_size=800):
    # training - 52287 for a closeup
    from utils import render_meshes, demo_color
    model_neutral = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', num_betas=11, use_pca=False, flat_hand_mean=True)

    dataset = AGORA_Dataset(
        split=split, force_build_dataset=0,
        res=res, extension=extension,
        training=training,
        img_size=img_size,
    )
    print(dataset)
    
    img_array, annot = dataset.__getitem__(i)

    img_array = denormalize_rgb(img_array, imagenet_normalization=1)
    verts_list = []
    for person in annot['humans']:
        with torch.no_grad():
            verts = model_neutral(
                global_orient=torch.from_numpy(person['smplx_root_pose']).reshape(1,-1),
                body_pose=torch.from_numpy(person['smplx_body_pose']).reshape(1,-1),
                jaw_pose=torch.from_numpy(person['smplx_jaw_pose']).reshape(1,-1),
                leye_pose=torch.from_numpy(person['smplx_leye_pose']).reshape(1,-1),
                reye_pose=torch.from_numpy(person['smplx_reye_pose']).reshape(1,-1),
                left_hand_pose=torch.from_numpy(person['smplx_left_hand_pose']).reshape(1,-1),
                right_hand_pose=torch.from_numpy(person['smplx_right_hand_pose']).reshape(1,-1),
                betas=torch.from_numpy(person['smplx_shape']).reshape(1,-1),
                transl=torch.from_numpy(person['smplx_transl']).reshape(1,-1),
                ).vertices.cpu().numpy().reshape(-1,3)
        verts_list.append(verts)
    faces_list = [model_neutral.faces for _ in annot['humans']]
    _color = [demo_color[0] for _ in annot['humans']]
    pred_rend_array = render_meshes(img_array.copy(), 
                                            verts_list,
                                            faces_list,
                                            {'focal': annot['K'][[0,1],[0,1]],
                                             'princpt': annot['K'][[0,1],[-1,-1]]},
                                            alpha=0.7,
                                            color=_color)
    img_array = np.concatenate([img_array, np.asarray(pred_rend_array)], 1)

    fn = f"{dataset.name}_{split}_{i}.jpg"
    Image.fromarray(img_array).save(fn)
    print(f"open {fn}")
    return 1
'''

def dataloader(split='validation', batch_size=4, num_workers=0, shuffle=1, img_size=512, n=-1, res=None):
    dataset = AGORA(
        split=split,
        img_size=img_size, 
        training=1, 
        n=n
    )

    print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn
    )

    for ii, (x, y) in enumerate(tqdm(dataloader)):
        sys.stdout.flush()
        if ii == 100:
            print()

#if __name__ == "__main__":
#    exec(sys.argv[1])

#dataloader(split='training', batch_size=4, num_workers=4, shuffle=1, img_size=512, n=-1, res=None)