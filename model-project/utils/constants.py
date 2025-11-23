# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
#
# Modified by: Dragos-Stefan Vacarasu, 2025
# Modifications: Added constants for dataset paths

import os

SMPLX_DIR = 'models'
MEAN_PARAMS = 'models/smpl_mean_params.npz'
CACHE_DIR_MULTIHMR = 'models/multiHMR'

ANNOT_DIR = 'data'
THREEDPW_DIR = 'data/3dpw'
BEDLAM_DIR = 'data/bedlam'
AGORA_DIR = 'data/agora'
AGORA_IMG_SIZE = (1280, 720)

SMPLX2SMPL_REGRESSOR = 'models/smplx/smplx2smpl.pkl'