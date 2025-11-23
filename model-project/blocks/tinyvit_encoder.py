# Copyright 2025 Dragos-Stefan Vacarasu
#
# This file was created as part of a modified version of Multi-HMR by NAVER Corp.
# The entire project is licensed under CC BY-NC-SA 4.0.

import torch
import timm

class TinyViTBackbone(torch.nn.Module):
    def __init__(self, name='tiny_vit_5m_224', pretrained=True, *args, **kwargs):
        super().__init__()
        self.name = name
        self.encoder = timm.create_model(name, pretrained=pretrained, *args, **kwargs)
        self.patch_size = 32
        self.embed_dim = 320

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.forward_features(x) # [bs, d, h, w]
        y = y.permute(0, 2, 3, 1) # [bs, h, w, d]
        y = y.reshape(y.shape[0], -1, y.shape[-1]) # [bs, h*w, d]
        return y