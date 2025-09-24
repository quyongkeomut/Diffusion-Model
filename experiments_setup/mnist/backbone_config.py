ENCODER_CONFIGS = {
    "img_channels": 3,
    "down_channels": (16, 32, 40),
    "expand_factor": 2,
    "num_groups_norm": 4,
    "activation": "hardswish",
    "dtype": None,
}

DECODER_CONFIGS = {
    "img_channels": 3,
    "latent_channels": 40,
    "up_channels": (32, 24, 16),
    "expand_factor": 2,
    "num_groups_norm": 4,
    "activation": "hardswish",
    "dtype": None,
}

DENOISE_MODEL_CONFIGS = {
    "img_channels": 3,
    "down_channels": (32, 64, 128),
    "is_conditional": None,
    "T": 100,  
    "t_embed_dim": 64,
    "expand_factor": 2,
    "num_groups_norm": 16,
    "activation": "gelu",
    "dtype": None,
}

def get_ae_configs(pretrained_encoder: bool = True):
    encoder_configs = {}
    if not pretrained_encoder:
        encoder_configs = ENCODER_CONFIGS
    decoder_configs = DECODER_CONFIGS
    
    return {
        "encoder": encoder_configs,
        "decoder": decoder_configs
    }
    

def get_denoise_model_configs(is_conditional: bool = True):
    configs = DENOISE_MODEL_CONFIGS
    configs["is_conditional"] = is_conditional
    if is_conditional:
        configs["kdim"] = 40
        configs["vdim"] = 40
    return configs