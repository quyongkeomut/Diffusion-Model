import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import random
import glob

import torch

from neural_nets.autoencoders.ae import encoder_mobilenet_v3_large, Encoder, Decoder
from neural_nets.denoise_models.unets import _UNetBase
from neural_nets.denoise_models.ddpm import DDPM, LatentDDPM

# from experiments_setup.pascal.backbone_config import get_ae_configs, get_denoise_model_configs
from experiments_setup.flowers102.backbone_config import get_ae_configs, get_denoise_model_configs


IS_CONDITIONAL = False
if IS_CONDITIONAL:
    path = "Diff_weights_conditional"
else:
    path = "Diff_weights"

CKPT_DENOISE_MODEL_DEFAULT = rf"./weights/{path}/flowers102/2025_01_20_08_47/denoise_model/denoise_model_last.pth"
IMG_SAVE_PATH = f"./weights/{path}/flowers102/2025_01_20_08_47/generated_images"

DENOISE_MODEL_CONFIGS = get_denoise_model_configs(img_channels=3, is_conditional=IS_CONDITIONAL)
rank = 0



#
# initialize the denoise model 
#    
# conditional encoder
# if IS_CONDITIONAL:
#     conditional_domain_encoder = encoder
# else:
#     conditional_domain_encoder = None    
    
# denoise model
model = _UNetBase(
    device=rank,
    **DENOISE_MODEL_CONFIGS
)
model_ckpt = torch.load(CKPT_DENOISE_MODEL_DEFAULT, weights_only=False)
model.load_state_dict(model_ckpt["model_state_dict"])

# DDPM
ddpm = DDPM(
    model=model,
    device=rank,
    **DENOISE_MODEL_CONFIGS
)


if __name__ == "__main__":
    for i in range(5):
     save_path = os.path.join(IMG_SAVE_PATH, f"translate_img_{i+1}")
     ddpm.show_translative_image(
         img_size=256, 
         save_path=save_path
     )