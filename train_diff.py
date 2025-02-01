import os
import time
import random
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from neural_nets.autoencoders.ae import encoder_mobilenet_v3_large, Encoder, Decoder
from neural_nets.denoise_models.unets import _UNetBase

from optimizer.optimizer import OPTIMIZERS

from augmentation.augmentation import CustomAug


NUM_DEVICE = torch.cuda.device_count()
CKPT_ENCODER_DEFAULT = "./weights/AEweights/flowers102/2025_01_24_15_52/encoder/encoder_last.pth"
CKPT_DECODER_DEFAULT = "./weights/AEweights/flowers102/2025_01_24_15_52/decoder/decoder_last.pth"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)


def ddp_setup(rank: int, world_size: int):
    """
    Init DDP

    Args:
        rank (int): A unique identifier that is assigned to each process
        world_size (int): Total process in a group
    """
    # this machine coordinates the communication across all processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12795"
    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    if rank == 0:
        time.sleep(30)


def main(
    rank: int,
    world_size: int,
    task: str,
    img_size,
    is_latent_diff,
    is_conditional,
    ckpt_encoder,
    ckpt_decoder,
    num_epochs,
    batch_size,
    check_point,
):
    ddp_setup(rank, world_size)
    
    if task == "flowers102":
        from datasets.flowers102 import Flowers102Diff
        from experiments_setup.flowers102.backbone_config import get_ae_configs, get_denoise_model_configs
        from experiments_setup.flowers102.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = Flowers102Diff(is_latent_diff=is_latent_diff, transform=CustomAug(img_size), split="train")
        VAL_DS = Flowers102Diff(is_latent_diff=is_latent_diff, transform=CustomAug(img_size), split="val")
        TEST_DS = Flowers102Diff(is_latent_diff=is_latent_diff, transform=CustomAug(img_size), split="test")
        TRAIN_DS = torch.utils.data.ConcatDataset([TRAIN_DS, VAL_DS, TEST_DS])

    elif task == "pascal":
        if is_conditional:
            from datasets.pascal import VOC2012Mask2Img
            TRAIN_DS = VOC2012Mask2Img(is_latent_diff=is_latent_diff, transform=CustomAug(img_size), size=img_size)
        else:
            from datasets.pascal import VOC2012Diff 
            TRAIN_DS = VOC2012Diff(is_latent_diff=is_latent_diff, transform=CustomAug(img_size), size=img_size)
        from experiments_setup.pascal.backbone_config import get_ae_configs, get_denoise_model_configs
        from experiments_setup.pascal.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        
    IS_PIN_MEMORY = True
    NUM_WORKERS = 2

    # these hyperparams depend on the dataset / experiment
    otim = OPTIMIZER_NAME
    optim_args = OPTIM_ARGS
    
    DENOISE_MODEL_CONFIGS = get_denoise_model_configs(
        img_channels=None if is_latent_diff else 3, 
        is_conditional=is_conditional
    ) 
    model = _UNetBase(
        device=rank,
        **DENOISE_MODEL_CONFIGS
    )    
    model.compile(fullgraph=True, backend="cudagraphs")
    
    if is_latent_diff:
        from neural_nets.denoise_models.ddpm import LatentDDPM
        from neural_nets.denoise_models.trainer import LatentDiffTrainer as Trainer
        
        # initialize the denoise model and optimizer
        AE_CONFIGS = get_ae_configs(pretrained_encoder=ckpt_encoder is None)
        decoder = Decoder(**AE_CONFIGS["decoder"], device=rank)
        decoder_ckpt = torch.load(ckpt_decoder, weights_only=False)
        decoder.load_state_dict(decoder_ckpt["model_state_dict"])

        if ckpt_encoder is not None:
            encoder = Encoder(**AE_CONFIGS["encoder"], device=rank)
            encoder_ckpt = torch.load(ckpt_encoder, weights_only=False)
            encoder.load_state_dict(encoder_ckpt["model_state_dict"])
            if is_conditional:
                conditional_domain_encoder = Encoder(**AE_CONFIGS["encoder"], device=rank)
                conditional_domain_encoder.load_state_dict(encoder_ckpt["model_state_dict"])
            else:
                conditional_domain_encoder = None
        else:
            encoder = encoder_mobilenet_v3_large().to(rank)
            if is_conditional:
                conditional_domain_encoder = encoder_mobilenet_v3_large(is_domain_encoder=True).to(rank)
            else:
                conditional_domain_encoder = None
            
        # Compile modules
        encoder.compile(fullgraph=True, backend="cudagraphs")
        decoder.compile(fullgraph=True, backend="cudagraphs")
        if is_conditional:
            conditional_domain_encoder.compile(fullgraph=True, backend="cudagraphs")
         

        ddpm = LatentDDPM(
            encoder=encoder,
            decoder=decoder,
            model=model,
            conditional_domain_encoder=conditional_domain_encoder,
            device=rank,
            **DENOISE_MODEL_CONFIGS
        )
        
    else:
        from neural_nets.denoise_models.ddpm import DDPM
        from neural_nets.denoise_models.trainer import DiffTrainer as Trainer
        ddpm = DDPM(
            model=model,
            device=rank,
            **DENOISE_MODEL_CONFIGS
        )
    
    optimizer = OPTIMIZERS[otim](
        [
            {"params": ddpm.model.parameters()},
        ],
        **optim_args
    )
    
    # print(optimizer)

    # load check point...
    if check_point:
        # load checkpoint
        check_point = torch.load(check_point, weights_only=False)
        
        # from checkpoint, load state_dict of optimizer and lr_schedulers
        # optimizer.load_state_dict(check_point["optimizer_state_dict"])
        # lr_scheduler_increase = LinearLR(
        #     optimizer,
        #     start_factor=1/5,
        #     total_iters=5
        # )
        # lr_scheduler_increase.load_state_dict(check_point["lr_increase_state_dict"])
        # lr_scheduler_cosine = CosineAnnealingLR(
        #     optimizer, 
        #     T_max=num_epochs-5,
        #     eta_min=1e-4
        # )
        # lr_scheduler_cosine.load_state_dict(check_point["lr_cosine_state_dict"])
        
        # # load the index of last training epoch
        # last_epoch = check_point["epoch"]
        encoder_state_dict = {k.replace("encoder.", ""): v for k, v in check_point["model_state_dict"].items() if k.startswith("encoder.")}
        decoder_state_dict = {k.replace("decoder.", ""): v for k, v in check_point["model_state_dict"].items() if k.startswith("decoder.")}

        # model.encoder.load_state_dict(encoder_state_dict)
        # model.decoder.load_state_dict(decoder_state_dict)
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None
        
    else:
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None

    # setup dataloaders
    train_loader = DataLoader(
        TRAIN_DS, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=DistributedSampler(TRAIN_DS), 
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=IS_PIN_MEMORY
    )
    # print(next(iter(train_loader))[0].shape)

    if is_latent_diff:
        weights_path_name = "LatentDiff_weights"
        if is_conditional:
            weights_path_name += "_conditional"
    else:
        weights_path_name = "Diff_weights"

    # out_dir = os.path.join(f"./weights/{weights_path_name}", task)
    out_dir = os.path.join(f"./weights/{weights_path_name}", task)
    
    # call the trainer
    criterion = MSELoss()
    trainer = Trainer(
        ddpm=ddpm,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        start_epoch=last_epoch,
        train_loader=train_loader,
        out_dir=out_dir,
        lr_scheduler_increase=lr_scheduler_increase,
        lr_scheduler_cosine=lr_scheduler_cosine,
        gpu_id=rank
    )
    
    trainer.fit()
    destroy_process_group()


if __name__ == "__main__":
    # environment setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCHDYNAMO_DYNAMIC_SHAPES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # The flags below controls whether to allow TF32 on cuda and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # arguments parser
    import argparse
    
    parser = argparse.ArgumentParser(description='Training args')

    parser.add_argument('--task', type=str, default="pascal", required=False, help='Dataset to train model on, valid values are one of [ade20k, pascal]')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--is_latent_diff', action='store_true', help='Decide if the model is the latent diffusion model or not')
    parser.add_argument('--is_conditional', action='store_true', help='Decide if the model is conditional generative model or not')
    parser.add_argument('--ckpt_encoder', nargs='?', const=CKPT_ENCODER_DEFAULT, default = None,
                        help='Checkpoint of pretrained encoder. If not been call, MobileNetv3 encoder will be used', )
    parser.add_argument('--ckpt_decoder', default=CKPT_DECODER_DEFAULT, help='Checkpoint of pretrained decoder')
    # parser.add_argument('--scale', type=float, default=0.25, required=False, help='Model scale')
    parser.add_argument('--epochs', type=int, default=20, help='Num epochs')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint for coutinue training')

    args = parser.parse_args()
    
    # setup model hyperparameters and training parameters
    task = args.task
    img_size = args.img_size
    is_latent_diff = args.is_latent_diff
    is_conditional = args.is_conditional
    ckpt_encoder = args.ckpt_encoder
    ckpt_decoder = args.ckpt_decoder
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    check_point = args.ckpt
    
    set_seed(seed)
    
    world_size = NUM_DEVICE
    args = (
        world_size, 
        task, 
        img_size, 
        is_latent_diff,
        is_conditional,
        ckpt_encoder,
        ckpt_decoder,
        num_epochs, 
        batch_size, 
        check_point
    )
    
    mp.spawn(main, args=args, nprocs=world_size)