# Copyright 2025 Dragos-Stefan Vacarasu
#
# This file was created as part of a modified version of Multi-HMR by NAVER Corp.
# The entire project is licensed under CC BY-NC-SA 4.0.

import unittest
import os
import torch
from PIL import Image, ImageOps
import numpy as np

from model_service import forward_model, load_model, infer_from_image
from demo import get_camera_parameters
from utils import normalize_rgb

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

class TestApp(unittest.TestCase):

    def setUp(self):
        self.ckpt_path = 'logs/threedpw/train_finetune_tinyvit_5m_5e-6/checkpoints/00000.pt'
        self.model = load_model(self.ckpt_path)

        self.test_image_file = "example_data/170149601_13aa4e4483_c.jpg"
        self.image_size = (672, 672)
        self.fov = 60

        self.image = Image.open(self.test_image_file)
        img_pil = ImageOps.contain(self.image, self.image_size)
        img_pil = ImageOps.pad(img_pil, self.image_size)

        resize_img = normalize_rgb(np.asarray(img_pil))
        self.x = torch.from_numpy(resize_img).unsqueeze(0).to(device)

        img_size = self.model.img_size
        self.camera_parameters = get_camera_parameters(img_size, self.fov)

    def test_forward_model_output(self):
        '''
        Test that the forward_model successfully runs an inference,
        and at least one person is detected.
        '''

        result = forward_model(self.model, self.x, self.camera_parameters)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], dict)
        self.assertIn("v3d", result[0])

    def test_infer_function(self):
        '''
        Test that the infer function successfully runs an inference
        and returns a list of result files.
        '''

        result_files = infer_from_image(self.model, self.image, 0.3, 60)

        self.assertIsInstance(result_files, list)
        self.assertEqual(len(result_files), 2)
        self.assertTrue(os.path.exists(result_files[0]))
        self.assertTrue(os.path.exists(result_files[1]))

if __name__ == "__main__":
    unittest.main()