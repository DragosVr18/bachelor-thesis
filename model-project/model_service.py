# Copyright 2025 Dragos-Stefan Vacarasu
#
# This file was created as part of a modified version of Multi-HMR by NAVER Corp.
# The entire project is licensed under CC BY-NC-SA 4.0.

import torch
import os
import time
import numpy as np
from PIL import Image, ImageOps

if torch.cuda.is_available() and torch.cuda.device_count()>0:
    device = torch.device('cuda:0')
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device_name = torch.cuda.get_device_name(0)
    print(f"Device - GPU: {device_name}")
else:
    device = torch.device('cpu')
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    device_name = 'CPU'
    print("Device - CPU")

from tinyvit_model import TinyViTModel
from demo import forward_model, get_camera_parameters, overlay_human_meshes
from utils import normalize_rgb, demo_color as color, create_scene
tmp_data_dir = 'tmp_data'

def load_model(ckpt_path, device=torch.device('cuda')):
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    kwargs['img_size'] = ckpt['args'].img_size
    model = TinyViTModel(**kwargs).to(device)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights loaded successfully")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

def infer_from_image(model, im, det_thresh, fov):
    global device

    if isinstance(im, np.ndarray):
        im = Image.fromarray(im.astype("uint8"))

    _basename = 'webcam_output'
    _glb_fn = f"{_basename}.glb"
    _rend_fn = f"{_basename}.png"
    glb_fn = os.path.join(tmp_data_dir, _glb_fn)
    rend_fn = os.path.join(tmp_data_dir, _rend_fn)
    os.makedirs(tmp_data_dir, exist_ok=True)

    fov, p_x, p_y = fov, None, None
    img_size = model.img_size

    K = get_camera_parameters(img_size, fov=fov, p_x=p_x, p_y=p_y, device=device)

    img_pil = ImageOps.contain(im, (img_size,img_size))

    width, height = img_pil.size
    pad = abs(width - height) // 2
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size, img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size, img_size))

    resize_img = normalize_rgb(np.asarray(img_pil))
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)

    img_array = np.asarray(img_pil_bis)
    img_pil_visu = Image.fromarray(img_array)

    start = time.time()
    humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=1)
    print(f"Forward: {time.time() - start:.2f}sec")

    start = time.time()
    pred_rend_array, _ = overlay_human_meshes(humans, K, model, img_pil_visu)
    rend_pil = Image.fromarray(pred_rend_array.astype(np.uint8))
    rend_pil.crop()
    if width > height:
        rend_pil = rend_pil.crop((0, pad, width, pad + height))
    else:
        rend_pil = rend_pil.crop((pad, 0, pad + width, height))
    rend_pil.save(rend_fn)
    print(f"Rendering with pyrender: {time.time() - start:.2f}sec")

    start = time.time()
    l_mesh = [humans[j]['v3d'].detach().cpu().numpy() for j in range(len(humans))]
    l_face = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]
    scene = create_scene(img_pil_visu, l_mesh, l_face, color=color, metallicFactor=0., roughnessFactor=0.5)
    scene.export(glb_fn)
    print(f"Exporting scene in glb: {time.time() - start:.2f}sec")

    return [rend_fn, glb_fn]