import tqdm
from contextlib import nullcontext
from datetime import datetime
import os
import csv
import glob
from typing import Type, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR


today = datetime.today()
formatted_today = today.strftime('%Y_%m_%d_%H_%M')


class DiffTrainer:
    def __init__(
        self,
        ddpm,
        criterion,
        optimizer,
        num_epochs: int,
        start_epoch: int = 0,
        train_loader = None,
        out_dir: str = "./weights/Diff_weights",
        lr_scheduler_increase: Optional[LinearLR] = None,
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        r"""
        _summary_

        Args:
            model (Type[Module]): _description_
            model_name (str): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
            num_epochs (int): _description_
            start_epoch (int, optional): _description_. Defaults to 0.
            train_loader (_type_, optional): _description_. Defaults to None.
            out_dir (str, optional): _description_. Defaults to "./weights/AEweights".
            lr_scheduler_increase (Optional[LinearLR], optional): _description_. Defaults to None.
            lr_scheduler_cosine (Optional[CosineAnnealingLR], optional): _description_. Defaults to None.
        """
        
        # setup model and ema model
        self.ddpm = ddpm 
        self.ddpm.model = self.ddpm.model.to(gpu_id) # this is required
        self.model = DDP(
            self.ddpm.model,
            device_ids=[gpu_id]
        )
        self.ema_denoise_model = AveragedModel(
            model=self.model.module, 
            multi_avg_fn=get_ema_multi_avg_fn(0.999)
        )
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer
        
        if lr_scheduler_increase is None: 
            self.lr_scheduler_increase = LinearLR(
                self.optimizer,
                start_factor=1/5,
                total_iters=5
            )
        else:
            self.lr_scheduler_increase = lr_scheduler_increase
        
        if lr_scheduler_cosine is None:
            self.lr_scheduler_cosine = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs-5,
                eta_min=1e-4
            )
        else:
            self.lr_scheduler_cosine = lr_scheduler_cosine
        
        # setup dataloaders
        self.train_loader = train_loader
        
        # setup training configs
        self.num_epochs = num_epochs
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.out_path = os.path.join(out_dir, formatted_today)
        self.denoise_model_out_path = os.path.join(self.out_path, "denoise_model")
        self.synthesis_images_path = os.path.join(self.out_path, "generated_images")
        os.makedirs(self.denoise_model_out_path, exist_ok=True)
        os.makedirs(self.synthesis_images_path, exist_ok=True)


    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss'])
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            
            if self.gpu_id == 0:
                # write results
                with open(train_csv_path, mode='a', newline='') as train_csvfile:
                    train_writer = csv.writer(train_csvfile)
                    train_writer.writerow(train_metrics)
                
                # Plot synthesis image and save them
                image_path = self.synthesis_images_path + f"{epoch + 1}.png"
                ...
            
            # lr schedulers step
            if epoch >= 5:
                self.lr_scheduler_cosine.step()
            else:
                self.lr_scheduler_increase.step()
            torch.cuda.empty_cache()

        # save EMA model
        if self.gpu_id == 0:
            save_path = os.path.join(self.denoise_model_out_path, f"denoise_model_ema.pth")
            torch.save({
                    "num_epoch": self.num_epochs,
                    "model_state_dict": self.ema_denoise_model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        
        # train_csvfile.close()
        # val_csvfile.close()
        

    def _run_one_step(self, data):
        # get data
        # T = self.ddpm.T
        # inputs: Tensor = data[0].to(self.gpu_id)
        #     
        # # generate data for each time step
        # times = torch.arange(0, T, device=self.gpu_id).unsqueeze(-1) # [T, 1]
        # with torch.no_grad():
        #     inputs = self.ddpm.encoder(inputs) # [N, C, H, W]
        #     
        #     inputs = inputs.unsqueeze(0).expand(T, -1, -1, -1, -1) ## [T, N, C, H, W], extend the time dimension            
        #     inputs, noises = self.ddpm.q_batched(inputs, times) # [T, N, C, H, W], [T, N, C, H, W]
        #     
        # if self.ddpm.is_conditional:
        #     self.ddpm.conditional_domain_encoder.eval()
        #     condition: Tensor = data[1].to(self.gpu_id) # [N, C, H, W]
        #     with torch.no_grad():
        #         condition = self.ddpm.conditional_domain_encoder(condition) # [N, C, H, W]
        #         condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [N, H*W, C]
        # 
        # in_dims = (0, 0)
        # if self.ddpm.is_conditional:
        #     in_dims = in_dims + (None, )
        #     predicted_noises =  torch.vmap(self.model.__call__, in_dims=in_dims, out_dims=0, randomness='different')(
        #         inputs,    
        #         times + 1,
        #         condition
        #     ) # [T, N, C, H, W]
        # else:
        #     predicted_noises =  torch.vmap(self.model.__call__, in_dims=in_dims, out_dims=0, randomness='different')(
        #         inputs,    
        #         times + 1,
        #     ) # [T, N, C, H, W]
        
        T = self.ddpm.T
        batch_size = data[0].shape[0]
        inputs: Tensor = data[0].to(self.gpu_id) # [N, C, H, W]
        
        # generate data for each time step
        times = torch.randint(0, T, (batch_size, 1), device=self.gpu_id).float() # [N, 1]
        inputs, noises = self.ddpm.q_batched(inputs, times) # [N, C, H, W], [N, C, H, W]
        
        predicted_noises = self.model(inputs, times) # [N, C, H, W]
        _loss = self.criterion(predicted_noises, noises) # [N, C, H, W]

        return _loss


    def train(self, epoch):
        self.model.train()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(f"TRAINING PHASE EPOCH: {epoch+1}")

        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                # clear gradient
                self.model.zero_grad(set_to_none=True)

                # perform 1 step
                _loss = self._run_one_step(data)
                
                # optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()                

                if self.gpu_id == 0:
                    # update ema model 
                    self.ema_denoise_model.update_parameters(self.model.module)
                    
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item())
                    pbar.update(1)  # Increase the progress bar

                    # save this step for backup...
                    save_path = os.path.join(self.denoise_model_out_path, "denoise_model_last.pth")

                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                        "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.train_loader))
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
        
        # also save model state dict along with optimizer's and scheduler's at the end of every epoch
        # os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
        # save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")
 
        # # os.makedirs(save_path, exist_ok=True)
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": self.model.state_dict(),
        #     "optimizer_state_dict": self.optimizer.state_dict(),
        #     "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
        #     "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
        #     "loss": loss,
        #     "metrics": metrics
        #     }, save_path)
 
        # # reset metrics tracker after every training epoch
        # self.metrics.reset()

        return [epoch + 1, f"{loss:4f}"]


    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()
        total_loss = 0
        print(F"VALIDATION PHASE EPOCH: {epoch+1}")
        with tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.val_loader:
                # get data
                inputs = data[0].to(self.gpu_id)
                targets = data[1]
                targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)

                targets = targets.permute(0, 3, 1, 2)
                
                targets = targets.to(self.gpu_id)
                
                # compute output, loss and metrics
                outputs = self.model(inputs)

                _loss = self.criterion(outputs, targets)

                # Convert to numpy
                outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                targets =  torch.argmax(targets, dim=1).cpu().detach().numpy()

                self.metrics.addBatch(outputs, targets)

                
                # calculate metrics of each task
                acc = self.metrics.pixelAccuracy()
                IoU = self.metrics.IntersectionOverUnion()
                mIoU = self.metrics.meanIntersectionOverUnion()


                metrics = {
                    "mIoU" : mIoU,
                    "IoU" : IoU,
                    "Acc" : acc
                }
                
                # update ema model 
                self.ema_model.update_parameters(self.model)

                # update progress bar
                total_loss += _loss.item()
                pbar.set_postfix(loss=_loss.item(), **metrics)
                pbar.update(1)  # Increase the progress bar
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.val_loader))
        print(f'Epoch {epoch+1} Loss: {loss:4f}')
        print()
        
        # save the best model on IoU metric
        current_IoU = mIoU 
        if current_IoU >= self.best_IoU:
            files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
            for file_path in files_to_delete:
                os.remove(file_path)

            save_path = os.path.join(self.out_path, f"best_IoU_{round(current_IoU,4)}_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                "loss": loss,
                "metrics": metrics
                }, save_path)
            
            self.best_IoU = current_IoU

        # reset metrics tracker after every validating epoch
        self.metrics.reset()
        
        return [epoch + 1, f"{loss:4f}", f"{acc:4f}", f"{IoU:4f}", f"{mIoU:4f}"]
    
   
class LatentDiffTrainer(DiffTrainer):
    def _run_one_step(self, data):
        self.ddpm.encoder.eval()
        T = self.ddpm.T
        batch_size = data[0].shape[0]
        inputs: Tensor = data[0].to(self.gpu_id) # [N, C, H, W]
        
        # generate data for each time step
        times = torch.randint(0, T, (batch_size, 1), device=self.gpu_id).float() # [N, 1]
        with torch.no_grad():
            inputs = self.ddpm.encoder(inputs) # [N, C, H, W]
            inputs, noises = self.ddpm.q_batched(inputs, times) # [N, C, H, W], [N, C, H, W]
        
        if self.ddpm.is_conditional:
            self.ddpm.conditional_domain_encoder.eval()
            condition: Tensor = data[1].to(self.gpu_id) # [N, C, H, W]
            with torch.no_grad():
                condition = self.ddpm.conditional_domain_encoder(condition) # [N, C, H, W]
                condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [N, H*W, C]
        else:
            condition = None
        
        predicted_noises = self.model(inputs, times, condition) # [N, C, H, W]
        _loss = self.criterion(predicted_noises, noises) # [N, C, H, W]

        return _loss   
 
    
class LatentDiffMNISTTrainer:
    def __init__(
        self,
        ddpm,
        criterion,
        optimizer,
        num_epochs: int,
        start_epoch: int = 0,
        train_loader = None,
        out_dir: str = "./weights/LatentDiffweights",
        lr_scheduler_increase: Optional[LinearLR] = None,
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        
        r"""
        _summary_

        Args:
            model (Type[Module]): _description_
            model_name (str): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
            num_epochs (int): _description_
            start_epoch (int, optional): _description_. Defaults to 0.
            train_loader (_type_, optional): _description_. Defaults to None.
            out_dir (str, optional): _description_. Defaults to "./weights/AEweights".
            lr_scheduler_increase (Optional[LinearLR], optional): _description_. Defaults to None.
            lr_scheduler_cosine (Optional[CosineAnnealingLR], optional): _description_. Defaults to None.
        """
        
        # setup model and ema model
        self.ddpm = ddpm 
        self.ddpm.model = self.ddpm.model.to(gpu_id) # this is required
        self.model = self.ddpm.model
        self.ema_denoise_model = AveragedModel(
            model=self.model, 
            multi_avg_fn=get_ema_multi_avg_fn(0.999)
        )
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer
        
        if lr_scheduler_increase is None: 
            self.lr_scheduler_increase = LinearLR(
                self.optimizer,
                start_factor=1/5,
                total_iters=5
            )
        else:
            self.lr_scheduler_increase = lr_scheduler_increase
        
        if lr_scheduler_cosine is None:
            self.lr_scheduler_cosine = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs-5,
                eta_min=1e-4
            )
        else:
            self.lr_scheduler_cosine = lr_scheduler_cosine
        
        # setup dataloaders
        self.train_loader = train_loader
        
        # setup training configs
        self.num_epochs = num_epochs
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.out_path = os.path.join(out_dir, formatted_today)
        self.denoise_model_out_path = os.path.join(self.out_path, "denoise_model")
        self.synthesis_images_path = os.path.join(self.out_path, "generated_images")
        os.makedirs(self.denoise_model_out_path, exist_ok=True)
        os.makedirs(self.synthesis_images_path, exist_ok=True)


    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss'])
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            
            if self.gpu_id == 0:
                # write results
                with open(train_csv_path, mode='a', newline='') as train_csvfile:
                    train_writer = csv.writer(train_csvfile)
                    train_writer.writerow(train_metrics)
                
                # Plot synthesis image and save them
                image_path = self.synthesis_images_path + f"{epoch + 1}.png"
                ...
            
            # lr schedulers step
            if epoch >= 5:
                self.lr_scheduler_cosine.step()
            else:
                self.lr_scheduler_increase.step()
            torch.cuda.empty_cache()

        # save EMA model
        if self.gpu_id == 0:
            save_path = os.path.join(self.denoise_model_out_path, f"denoise_model_ema.pth")
            torch.save({
                    "num_epoch": self.num_epochs,
                    "model_state_dict": self.ema_denoise_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        
        # train_csvfile.close()
        # val_csvfile.close()
        

    def _run_one_step(self, data):
        # get data
        # T = self.ddpm.T
        # inputs: Tensor = data[0].to(self.gpu_id)
        #     
        # # generate data for each time step
        # times = torch.arange(0, T, device=self.gpu_id).unsqueeze(-1) # [T, 1]
        # with torch.no_grad():
        #     inputs = self.ddpm.encoder(inputs) # [N, C, H, W]
        #     
        #     inputs = inputs.unsqueeze(0).expand(T, -1, -1, -1, -1) ## [T, N, C, H, W], extend the time dimension            
        #     inputs, noises = self.ddpm.q_batched(inputs, times) # [T, N, C, H, W], [T, N, C, H, W]
        #     
        # if self.ddpm.is_conditional:
        #     self.ddpm.conditional_domain_encoder.eval()
        #     condition: Tensor = data[1].to(self.gpu_id) # [N, C, H, W]
        #     with torch.no_grad():
        #         condition = self.ddpm.conditional_domain_encoder(condition) # [N, C, H, W]
        #         condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [N, H*W, C]
        # 
        # in_dims = (0, 0)
        # if self.ddpm.is_conditional:
        #     in_dims = in_dims + (None, )
        #     predicted_noises =  torch.vmap(self.model.__call__, in_dims=in_dims, out_dims=0, randomness='different')(
        #         inputs,    
        #         times + 1,
        #         condition
        #     ) # [T, N, C, H, W]
        # else:
        #     predicted_noises =  torch.vmap(self.model.__call__, in_dims=in_dims, out_dims=0, randomness='different')(
        #         inputs,    
        #         times + 1,
        #     ) # [T, N, C, H, W]
        
        T = self.ddpm.T
        batch_size = data[0].shape[0]
        inputs: Tensor = data[0].to(self.gpu_id) # [N, C, H, W]
        
        # generate data for each time step
        times = torch.randint(0, T, (batch_size,), device=self.gpu_id).unsqueeze(-1).float() # [N, 1]
        # with torch.no_grad():
        #     inputs = self.ddpm.encoder(inputs) # [N, C, H, W]
        #     inputs, noises = self.ddpm.q_batched(inputs, times) # [N, C, H, W], [N, C, H, W]
        inputs, noises = self.ddpm.q_batched(inputs, times) # [N, C, H, W], [N, C, H, W]
                
        if self.ddpm.is_conditional:
            self.ddpm.conditional_domain_encoder.eval()
            condition: Tensor = data[1].to(self.gpu_id) # [N, C, H, W]
            with torch.no_grad():
                condition = self.ddpm.conditional_domain_encoder(condition) # [N, C, H, W]
                condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [N, H*W, C]
        else:
            condition = None
        
        predicted_noises = self.model(inputs, times, condition) # [N, C, H, W]
        _loss = self.criterion(predicted_noises, noises) # [N, C, H, W]

        return _loss


    def train(self, epoch):
        self.model.train()
        self.ddpm.encoder.eval()
        total_loss = 0
        
        if self.gpu_id == 0:
            print(f"TRAINING PHASE EPOCH: {epoch+1}")

        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                # clear gradient
                self.model.zero_grad(set_to_none=True)

                # perform 1 step
                _loss = self._run_one_step(data)
                
                # optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()                

                if self.gpu_id == 0:
                    # update ema model 
                    self.ema_denoise_model.update_parameters(self.model)
                    
                    # update progress bar
                    total_loss += _loss.item()
                    pbar.set_postfix(loss=_loss.item())
                    pbar.update(1)  # Increase the progress bar

                    # save this step for backup...
                    save_path = os.path.join(self.denoise_model_out_path, "denoise_model_last.pth")

                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
                        "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = (total_loss/len(self.train_loader))
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
        
        # also save model state dict along with optimizer's and scheduler's at the end of every epoch
        # os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
        # save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")
 
        # # os.makedirs(save_path, exist_ok=True)
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": self.model.state_dict(),
        #     "optimizer_state_dict": self.optimizer.state_dict(),
        #     "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
        #     "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
        #     "loss": loss,
        #     "metrics": metrics
        #     }, save_path)
 
        # # reset metrics tracker after every training epoch
        # self.metrics.reset()

        return [epoch + 1, f"{loss:4f}"]
    