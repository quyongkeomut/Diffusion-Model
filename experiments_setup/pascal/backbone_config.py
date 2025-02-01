ENCODER_CONFIGS = {
    "img_channels": 3,
    "down_channels": (16, 24),
    "expand_factor": 2,
    "num_groups_norm": 4,
    "activation": "swish",
    "dtype": None,
}

DECODER_CONFIGS = {
    "img_channels": 3,
    "latent_channels": 24,
    "up_channels": (24, 16),
    "expand_factor": 2,
    "num_groups_norm": 4,
    "activation": "swish",
    "dtype": None,
}

DENOISE_MODEL_CONFIGS = {
    "img_channels": None,
    "down_channels": (64, 128, 256),
    "is_conditional": None,
    "T": 400,  
    "t_embed_dim": 32,
    "expand_factor": 2,
    "num_groups_norm": 8,
    "activation": "swish",
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
    

def get_denoise_model_configs(
    img_channels: int | None,
    is_conditional: bool
):
    configs = DENOISE_MODEL_CONFIGS
    if img_channels is None:
        configs["img_channels"] = ENCODER_CONFIGS["down_channels"][-1]
    else:
        configs["img_channels"] = img_channels
    configs["is_conditional"] = is_conditional
    if is_conditional:
        configs["kdim"] = 24
        configs["vdim"] = 24
    return configs