from typing import Optional, Dict, Tuple, Any
import argparse
import os
import time
import random
import yaml
import numpy as np

import torch
from torch.nn import Module
from torch.distributed import init_process_group

from neural_nets.denoise_models import *
from losses import *
from optimizer.optimizer import OPTIMIZERS
from .trainers import *
import datasets


DDPM_CRITERION_TRAINER = {
    'DDPM': (
        ddpm.DDPM,
        unets.BaseUNet,
        ELBO.ELBOLoss,
        base_trainer.BaseTrainer 
    ),
    'LatentDiff': (
        ...
    ),
}


def get_args_argparse() -> Tuple[Dict, Dict]:
    """
    Getting arguments parser method

    Returns:
        Dict: A dictionary contains all keyword arguments for experiment.
    """
    def parse_kwargs(args):
        kwargs = {}
        key = None
        for arg in args:
            if arg.startswith('--'):
                key = arg.lstrip('--')
                kwargs[key] = True  # default to flag
            elif key:
                kwargs[key] = arg
                key = None
        return kwargs
    
    parser = argparse.ArgumentParser(description='Training args')

    # model type
    parser.add_argument(
        '--model', type=str, default="VAE", required=False, 
        help=f'Type of model, valid values are one of {list(DDPM_CRITERION_TRAINER.keys())}'
    )
    
    # dataset
    parser.add_argument(
        '--dataset', type=str, default="mnist", required=False, 
        help=f'Dataset to train model on, valid values are one of {datasets.__all__}'
    )
    
    # DDP (Distributed Data Parallel) option
    parser.add_argument(
        '--is_ddp', action="store_true", 
        help='Option for choosing training in DDP or normal training criteria'
    )
    
    # conditional DDPM
    parser.add_argument(
        '--is_conditional', action='store_true', 
        help='Decide if the model is conditional generative model or not'
    )
    
    # image size
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=50, help='Num epochs')
    
    # Batch size
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    # Learning rate
    parser.add_argument(
        '--lr', type=float, default=1e-3, 
        help='Learning rate'
    )
    
    # Reconstruction method
    parser.add_argument(
        '--recon', type=str, default='mse', 
        help='Reconstruction method, valid values are one of ["mse", "bce"]'
    )
    
    # Random seed
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for training'
    )
    
    # Saving directory
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default="./weights", 
        help='Directory to save weights and all results'
    )
    
    base_args, model_args = parser.parse_known_args()
    return vars(base_args), parse_kwargs(model_args)


def get_modules_criterion_trainer(
    model_type: str,
    denoise_model_configs: Dict,
    backend: Optional[str] = None,
    *args,
    **kwargs
) -> Tuple[Module, base_trainer.BaseTrainer, Any]:
    # load type of Diffusion-based model
    try:
        Diff_model, denoise_model, criterion, trainer = DDPM_CRITERION_TRAINER[model_type]
        Diff_model = Diff_model(denoise_model, *args, **denoise_model_configs, **kwargs)
    except KeyError:
        raise KeyError(
            f"Model must be one of {list(DDPM_CRITERION_TRAINER.keys())}, got {model_type} instead"
        )
        
    # Compile modules
    if backend is not None:
        Diff_model.compile(fullgraph=True, backend=backend)
    return Diff_model, criterion, trainer
    
    
def get_optimizer(
    optim_name: str,
    *modules,
    **optim_kwargs
) -> ...:
    optimizer = OPTIMIZERS[optim_name](
        [
            {"params": module.parameters()}
            for module in modules
        ],
        **optim_kwargs,
    )
    return optimizer
    
    
def set_seed(seed: int) -> None:
    """
    Setting random seed method

    Args:
        seed (int): Seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    

def set_ddp(rank: int, world_size: int) -> None:
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
        

def set_env() -> None:
    """
    Method for setting other enviroment variables
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for debugging
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
    

def load_configs(model_type: str) -> Dict:
    filepath = f"./experiments_setup/configs/{model_type}.yaml"
    try:
        with open(filepath, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        print(f"Model must be one of {list(DDPM_CRITERION_TRAINER.keys)}, got {model_type} instead")
    return config