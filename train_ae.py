import os
import time
import random
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from neural_nets.autoencoders.trainer import AETrainer
from neural_nets.autoencoders.ae import encoder_mobilenet_v3_large, Encoder, Decoder

from optimizer.optimizer import OPTIMIZERS

from augmentation.augmentation import CustomAug


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
    os.environ["MASTER_PORT"] = "12397"
    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    if rank == 0:
        time.sleep(30)


NUM_DEVICE = torch.cuda.device_count()


def main(
    rank: int,
    world_size: int,
    task: str,
    img_size: int | tuple,
    pretrained_encoder: bool,
    num_epochs: int,
    batch_size: int,
    ckpt_encoder,
    ckpt_decoder
):
    ddp_setup(rank, world_size)
    
    if task == "flowers102":
        from datasets.flowers102 import Flowers102AE
        from experiments_setup.flowers102.backbone_config import get_ae_configs
        from experiments_setup.flowers102.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        BACKBONE_CONFIGS = get_ae_configs(pretrained_encoder)
        IS_PIN_MEMORY = True
        NUM_WORKERS = 2
        TRAIN_DS = Flowers102AE(transform=CustomAug(img_size), split="train")
        VAL_DS = Flowers102AE(transform=CustomAug(img_size), split="val")
        TEST_DS = Flowers102AE(transform=CustomAug(img_size), split="test")
        TRAIN_DS = torch.utils.data.ConcatDataset([TRAIN_DS, VAL_DS, TEST_DS])

    elif task == "pascal":
        from datasets.pascal import VOC2012AE
        from experiments_setup.pascal.backbone_config import get_ae_configs
        from experiments_setup.pascal.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        BACKBONE_CONFIGS = get_ae_configs(pretrained_encoder)
        IS_PIN_MEMORY = True
        NUM_WORKERS = 2
        TRAIN_DS = VOC2012AE(transform=CustomAug(img_size), size=img_size)
    
    out_dir = os.path.join("./weights/AEweights", task)

    # these hyperparams depend on the dataset / experiment
    otim = OPTIMIZER_NAME
    optim_args = OPTIM_ARGS

    # initialize the encoder, decoder and optimizer
    decoder = Decoder(**BACKBONE_CONFIGS["decoder"], device=rank)
    if not pretrained_encoder:
        encoder = Encoder(**BACKBONE_CONFIGS["encoder"], device=rank)
        optimizer = OPTIMIZERS[otim](
            [
                {"params": encoder.parameters()},
                {"params": decoder.parameters()},
            ],
            **optim_args
        )
    else:
        encoder = encoder_mobilenet_v3_large().to(rank)
        optimizer = OPTIMIZERS[otim](
            [
                {"params": decoder.parameters()},
            ],
            **optim_args
        )
    
    # print(optimizer)

    # load check point...
    if ckpt_encoder:
        # load checkpoint
        ckpt_encoder = torch.load(ckpt_encoder, weights_only=False)
        
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
        # encoder_state_dict = {k.replace("encoder.", ""): v for k, v in check_point["model_state_dict"].items() if k.startswith("encoder.")}
        # decoder_state_dict = {k.replace("decoder.", ""): v for k, v in check_point["model_state_dict"].items() if k.startswith("decoder.")}

        encoder.load_state_dict(ckpt_encoder["model_state_dict"])
        # model.decoder.load_state_dict(decoder_state_dict)
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None
        
    else:
        last_epoch = 0
        lr_scheduler_increase = None
        lr_scheduler_cosine = None
        
    if ckpt_decoder:
        ckpt_decoder = torch.load(ckpt_decoder, weights_only=False)
        decoder.load_state_dict(ckpt_decoder["model_state_dict"])
        
    # Compile modules
    encoder.compile(fullgraph=True, backend="cudagraphs")
    decoder.compile(fullgraph=True, backend="cudagraphs")   

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

    # call the trainer
    criterion = torch.nn.MSELoss()
    trainer = AETrainer(
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        start_epoch=last_epoch,
        save_encoder=not pretrained_encoder,
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
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--pretrained_encoder', action='store_true', help='Use pretrained encoder or not')
    # parser.add_argument('--scale', type=float, default=0.25, required=False, help='Model scale')
    parser.add_argument('--epochs', type=int, default=10, help='Num epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    parser.add_argument('--ckpt_encoder', nargs='?', default = None,
                        help='Checkpoint of pretrained encoder for coutinue training')
    parser.add_argument('--ckpt_decoder', nargs='?', default = None,
                        help='Checkpoint of pretrained decoder for coutinue training')

    args = parser.parse_args()
    
    # setup model hyperparameters and training parameters
    task = args.task
    img_size = args.img_size
    pretrained_encoder = args.pretrained_encoder
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    ckpt_encoder = args.ckpt_encoder
    ckpt_decoder = args.ckpt_decoder
    set_seed(seed)
    
    world_size = NUM_DEVICE
    args = (
        world_size, 
        task, 
        img_size, 
        pretrained_encoder, 
        num_epochs, 
        batch_size, 
        ckpt_encoder,
        ckpt_decoder
    )
    mp.spawn(main, args=args, nprocs=world_size)