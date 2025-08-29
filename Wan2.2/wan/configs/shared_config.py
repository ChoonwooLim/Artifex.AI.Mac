# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = ('vivid colors, static, blurry details, subtitles, style, artwork, painting, picture, still, overall gray, worst '
                                     'quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, '
                                     'poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused '
                                     'fingers, static image, cluttered background, three legs, crowded background, walking '
                                     'backwards')
wan_shared_cfg.frame_num = 81
