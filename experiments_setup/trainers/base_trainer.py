from typing import Optional, Any, Dict, Tuple
import tqdm
from datetime import datetime
import os
import csv
import time
import random
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.other_utils import save_animation

TODAY = datetime.today()


class BaseTrainer:

    METRIC_KEYS = ['Loss']
    AVG_WEIGHT = 0.99
    ANIMATION_SIZE = 32
    FORMATTED_TODAY = TODAY.strftime('%Y_%m_%d_%H_%M')
    
    def __init__(
        self,
        is_ddp: bool,
        ddpm: Module,
        base_lr: float,
        criterion: Any,
        optimizer: Any,
        num_epochs: int,
        train_loader: Any,
        save_dir: str | Path,
        start_epoch: int = 0,
        lr_scheduler: Optional[Any] = None,
        gpu_id: int = 0,
        gradient_accumulation_steps: int = 3,
        *args,
        **kwargs
    ) -> None:
        """
        Generic Diffusion Model Trainer

        Args:
            is_ddp (bool): Decide whether the training setting is DDP or normal criteria. 
            ddpm (Type[Module]): DDPM.
            base_lr (float): Baseline learning rate.
            criterion (_type_): Loss function.
            optimizer (_type_): Parameter optimizer.
            num_epochs (int): Number of training epochs.
            train_loader (_type_, optional): Decide whether saving the decoder. Defaults to None.
            save_dir (str, optional): Folder to save the weights.
            start_epoch (int, optional): Start epoch index. Defaults to 0.
            lr_scheduler (Optional[CosineAnnealingLR], optional): Learning rate scheduler. 
                Defaults to None.
            gpu_id (int, optional): GPU index. Defaults to 0.
            gradient_accumulation_steps (int, optional): Steps to accumulate gradient. Defaults to 3.
        """
        self.is_ddp = is_ddp 
        self.gpu_id = gpu_id
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        
        # setup model and ema model
        self.ddpm = ddpm.to(gpu_id)
        self.T = ddpm.T
        self._setup_module()
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._setup_lr_scheduler()
        
        # setup dataloader(s)
        self.train_loader = train_loader
        
        # setup paths to save model
        self.out_path = os.path.join(save_dir, self.FORMATTED_TODAY)
        os.makedirs(self.out_path, exist_ok=True)
        
        # setup path to save results
        self.val_out_path = os.path.join(self.out_path, "val_results")
        os.makedirs(self.val_out_path, exist_ok=True)
            
        
    def _setup_module(self) -> None:
        if self.is_ddp:
            self.ddpm: DDP = DDP(
                self.ddpm,
                device_ids=[self.gpu_id]
            )
            
        if self.gpu_id == 0:
            self.ema_ddpm = AveragedModel(
                model=self.ddpm.module if self.is_ddp else self.ddpm, 
                multi_avg_fn=get_ema_multi_avg_fn(self.AVG_WEIGHT)
            )
    
    
    def _setup_lr_scheduler(self) -> None:
        if self.lr_scheduler is None:
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs,
                eta_min=self.base_lr/2
            )
                
    
    def _save_modules(
        self, 
        epoch,
        save_path: str, 
        *state_dicts
    ):
        module_state_dict, optim_state_dict, scheduler_state_dict = state_dicts
        torch.save({
            "num_epoch": epoch,
            "model_state_dict": module_state_dict,
            "optimizer_state_dict": optim_state_dict,
            "lr_scheduler_state_dict": scheduler_state_dict,
        }, save_path)
    
    
    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        # FOR LOGGING
        
        self._on_train_begin()
        
        # FITTING LOOP
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            # Train
            self._train(epoch)
            # validating
            if (epoch + 1) % 5 == 0 and self.gpu_id == 0:
                self._validate(epoch)
                
        # AT THE END OF TRAINING
        self._on_train_end()


    def _on_train_begin(self):
        train_csv_path = os.path.join(self.out_path, 'train_metrics.csv')
        with open(train_csv_path, mode='w+', newline='') as train_csvfile:
            train_writer = csv.writer(train_csvfile)
            train_writer.writerow(['Epoch'] + self.METRIC_KEYS)
        self.train_csv_path = train_csv_path
        
        # FOR SAVING RESULTS FROM BEGINING TO THE END
        self.all_results = []
        self.all_results_ema = []
        
        
    def _on_train_end(self):
        # clear gradient
        self.ddpm.zero_grad(set_to_none=True)
        
        if self.gpu_id == 0:
             # save EMA model
            save_path = os.path.join(self.out_path, f"ddpm_ema_model.pth")
            ddpm_state_dicts = (
                self.ema_ddpm.module.state_dict(), 
                self.optimizer.state_dict(), 
                self.lr_scheduler.state_dict()
            )
            self._save_modules(self.num_epochs, save_path, *ddpm_state_dicts)
        
            # save all generated images
            res_save_path = os.path.join(self.out_path, f"All_Evaluation_Results.gif")
            res_ema_save_path = os.path.join(self.out_path, f"All_Evaluation_Results_EMA.gif")
            save_animation(
                self.all_results, 
                res_save_path, 
                interval=100, 
                repeat_delay=3000
            )
            save_animation(
                self.all_results_ema, 
                res_ema_save_path, 
                interval=100, 
                repeat_delay=3000
            )
        
    
    def _forward_pass(self, data: Tuple[Tensor, Any]) -> Dict[str, Tensor]:            
        # get data
        batch_size = data[0].shape[0]
        inputs: Tensor = data[0].to(self.gpu_id) # shape (N, C, H, W)
        
        # generate data for each time step
        times: Tensor = torch.randint(1, self.T+1, size=(batch_size,), device=self.gpu_id).long() # shape (N, ); value range [1, T]
        if self.is_ddp:
            inputs_T = self.ddpm.module.q(inputs, times) 
        else:
            inputs_T = self.ddpm.q(inputs, times) # (N, C, H, W)
        
        # compute output, loss and metrics
        X_pred = self.ddpm(inputs_T, times) # (N, C, H, W)
        _losses = self.criterion(self.ddpm, times, X_pred, inputs, is_ddp=self.is_ddp)

        return _losses

    
    def _optimize_step(
        self, 
        idx_data: int,
        data: Tuple[Tensor, Any]
    ) -> Dict[str, Tensor]:
        # Forward pass
        _losses = self._forward_pass(data)
        
        # Backward pass
        _loss = _losses["Loss"]
        _loss.backward(retain_graph=True)
        
        # Accumulate gradient and Optimize step
        if self._current_gradient_step % self.gradient_accumulation_steps == 0 or idx_data == len(self.train_loader) - 1:
            self.optimizer.step()
            
            # clear gradient
            self.ddpm.zero_grad(set_to_none=True)
            
        self._current_gradient_step += 1
        return _losses
    

    def _on_train_epoch_begin(self, epoch: int) -> None:
        # clear gradient
        self.ddpm.zero_grad(set_to_none=True)
        self.ddpm.train()
        self._current_gradient_step = 1
        
        if self.gpu_id == 0:
            print(f"TRAINING PHASE - EPOCH {epoch+1}")
        time.sleep(2)
    
    
    def _on_train_epoch(
        self, 
        epoch: int, 
    ) -> Dict[str, float]:
        
        _metrics = {
            metric_key: None for metric_key in self.METRIC_KEYS
        }
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for i, data in enumerate(self.train_loader):
                
                # Forward pass and Optimization step
                _losses = self._optimize_step(i, data)
                
                # Update metrics and progress bar
                if self.gpu_id == 0:
                    # update progress bar
                    for metric_key, current_metric in _metrics.items():
                        if current_metric is not None:
                            numeric_metric = float(current_metric)
                            _metrics[metric_key] = f"{0.9*numeric_metric + 0.1*_losses[metric_key].detach().item():.4f}"
                        else:
                            _metrics[metric_key] = f"{_losses[metric_key].detach().item():.4f}"
                    pbar.set_postfix(_metrics)
                    pbar.update(1)
                    
                    # Update EMA modules 
                    if self.gpu_id == 0:
                        if self.is_ddp: 
                            self.ema_ddpm.update_parameters(self.ddpm.module)
                        else:
                            self.ema_ddpm.update_parameters(self.ddpm)
                    
                    # Save this step for backup...
                    save_path = os.path.join(self.out_path, "ddpm_last.pth")
                    state_dicts = (
                        self.optimizer.state_dict(), 
                        self.lr_scheduler.state_dict()
                    )
                    if self.is_ddp:
                        ddpm_state_dicts = (self.ddpm.module.state_dict(),) + state_dicts
                    else:
                        ddpm_state_dicts = (self.ddpm.state_dict(),) + state_dicts
                    self._save_modules(epoch, save_path, *ddpm_state_dicts)

                
                # clear cache
                torch.cuda.empty_cache()

        # calculate averaged loss
        return _metrics
    
    
    def _on_train_epoch_end(
        self, 
        epoch: int,
        train_metrics: Dict[str, float| None],
    ):
        if self.gpu_id == 0:
            _train_metrics_list = [epoch + 1]
            for metric_key in self.METRIC_KEYS:
                _train_metrics_list.append(
                    float(train_metrics[metric_key]) if train_metrics[metric_key] is not None else 0
                )
            
            # write results
            with open(self.train_csv_path, mode='a', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(_train_metrics_list)
        # lr schedulers step
        self.lr_scheduler.step()
        torch.cuda.empty_cache()
        time.sleep(2)
    
    
    def _train(self, epoch: int):
        self._on_train_epoch_begin(epoch)
        _metrics = self._on_train_epoch(epoch)
        self._on_train_epoch_end(epoch, _metrics)

        
    def _on_val_epoch_begin(self):
        self.ddpm.eval()
        self.ema_ddpm.eval()
    
    
    def _on_val_epoch(self, epoch: int):
        save_path = os.path.join(self.val_out_path, f"Evaluation result - EPOCH {epoch+1}.gif")
        save_path_ema = os.path.join(self.val_out_path, f"Evaluation result EMA - EPOCH {epoch+1}.gif")
        if self.is_ddp:
            results = self.ddpm.module.sample(self.ANIMATION_SIZE)
        else:
            results = self.ddpm.sample(self.ANIMATION_SIZE)
        results_ema = self.ema_ddpm.module.sample(self.ANIMATION_SIZE)
        
        self.all_results.extend(random.sample(results, self.ANIMATION_SIZE//2))
        self.all_results_ema.extend(random.sample(results_ema, self.ANIMATION_SIZE//2))
        save_animation(results, save_path)
        save_animation(results_ema, save_path_ema)
            
    
    def _on_val_epoch_end(self, epoch: int):
        print(f"EVALUATION STEP - EPOCH {epoch+1} - DONE")
        time.sleep(2)
    
    
    @torch.no_grad()
    def _validate(self, epoch):
        self._on_val_epoch_begin()
        self._on_val_epoch(epoch)
        self._on_val_epoch_end(epoch)
  