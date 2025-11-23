# Copyright 2025 Dragos-Stefan Vacarasu
#
# This file was created as part of a modified version of Multi-HMR by NAVER Corp.
# The entire project is licensed under CC BY-NC-SA 4.0.

from model import Model, HPH, regression_mlp, FourierPositionEncoding
from blocks import TinyViTBackbone

class TinyViTModel(Model):
    """
    Extends the Model class to replace the backbone with TinyViTBackbone.
    """
    def __init__(self,
                 pretrained_backbone: bool = False,
                 *args, **kwargs):
        super().__init__(
            load_backbone=False,  
            load_hph=False,
            *args,
             **kwargs)

        self.backbone = TinyViTBackbone(pretrained=pretrained_backbone)
        self.embed_dim  = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        if self.camera_embedding is not None:
            if not self.camera_embedding == 'geometric':
                raise NotImplementedError("Only geometric camera embedding is implemented")
            self.camera = FourierPositionEncoding(n=3, num_bands=self.num_bands,max_resolution=self.max_resolution)
            self.camera_embed_dim = self.camera.channels

        self.mlp_classif = regression_mlp([self.embed_dim, self.embed_dim, 1])
        self.mlp_offset  = regression_mlp([self.embed_dim, self.embed_dim, 2])

        self.x_attention_head = HPH(
            num_body_joints=self.nrot - 1,
            context_dim=self.embed_dim + self.camera_embed_dim,
            dim=384,
            depth=self.xat_depth,
            heads=self.xat_num_heads,
            mlp_dim=384,
            dim_head=24,
            dropout=0.0,
            emb_dropout=0.0,
            at_token_res=self.img_size // self.patch_size,
            num_betas=self.num_betas,
        )

