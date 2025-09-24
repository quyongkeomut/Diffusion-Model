from pathlib import Path
from typing import (
    Tuple, 
    Type,
    Optional
)

import numpy as np

import torch
from torch import (
    Tensor,
    linspace,
    cumprod,
    sqrt,
    randn,
    empty,
    randn_like,
    sigmoid,
    vmap
)
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils.other_utils import to_image


class DDPM(Module):
    def __init__(
        self,
        denoise_model: Type[Module],
        T: int,
        beta: float = 0.25,
        **kwargs
    ) -> None:
        super().__init__()
        self.img_channels = kwargs['img_channels']
        self.T = T
        self.device = kwargs['device']
        self.model = denoise_model(**kwargs)
        self.reconstruction_method = kwargs['reconstruction_method']
        
        betas = linspace(0.0, beta, steps=self.T+1, device=self.device).view(self.T+1, 1, 1, 1) # shape (T+1, 1, 1, 1); T time step 0 -> T, 
        alphas = 1.0 - betas # (T+1, 1, 1, 1); [1, ..., 0.77]
        alphas_bar = cumprod(alphas, dim=0) # (T+1, 1, 1, 1) ; [1, ..., near 0]
        one_minus_alphas_bar = 1.0 - alphas_bar # (T+1, 1, 1, 1); [0, ... near 1]
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = sqrt(alphas_bar)
        self.square_betas = betas**2 # (T+1, 1, 1, 1)
        self.one_minus_alphas_bar = one_minus_alphas_bar
        self.square_one_minus_alphas_bar = one_minus_alphas_bar**2
        self.sqrt_one_minus_alphas_bar = sqrt(one_minus_alphas_bar)
        
        # foward diffusion hyperparameters
        self.fw_mean_coef = self.sqrt_alphas_bar # (T+1, 1, 1, 1)
        self.fw_std = self.sqrt_one_minus_alphas_bar # (T+1, 1, 1, 1)
        
        # reverse diffusion hyperparameters
        self.x_previous_coef = sqrt(alphas) / one_minus_alphas_bar # (T+1, 1, 1, 1)
        self.x0_pred_coef = betas / one_minus_alphas_bar # (T+1, 1, 1, 1)
        self.bw_std = sqrt(betas / one_minus_alphas_bar)
        
        # forward diff batched function
        # self.q_batched = vmap(
        #     self.q, 
        #     in_dims=0,
        #     out_dims=0, 
        #     randomness="different"
        # )
        # 
        # # sampling (reverse diff) batched function
        # self.reverse_q_batched = vmap(
        #     self.reverse_q, 
        #     in_dims=0, 
        #     out_dims=0,
        #     randomness="different"
        # )
        
        
    @torch.no_grad
    def q(
        self, 
        x_0: Tensor, 
        T: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward diffusion process

        Args:
            x_0 (Tensor): Initial latent
            T (int): Time step at t 

        Shape:
            x_0: (N, C, H, W)
            T: (N, )
        
        Returns:
            Tensor (N, C, H, W): x_t, which is the added noise version of x_0 after t steps
            Tensor (N, C, H, W): the noise added at step t
        """
        T = T.int() # (N, )
        noise = randn_like(x_0) # (N, C, H, W)
        mean = self.fw_mean_coef[T] * x_0 # (N, C, H, W)
        x_t =  mean + self.fw_std[T] * noise
        return x_t
        

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
        
    @torch.no_grad    
    def reverse_q(
        self,
        x_t: Tensor, 
        x0_predicted: Tensor,
        T: Tensor,
    ) -> Tensor:
        """
        Reverse diffusion process. x_t-1 is sampled based on current x_t,
        current time step t and predicted noise to produce x_t

        Args:
            x_t (Tensor): current sample at time step t
            T (Tensor): Time step at t
            noise_predicted (Tensor): the noise that produce x_t


        Shape:
            x_t: (N, C, H, W)
            T: (N,)
            noise_predicted: (N, C, H, W)

        Returns:
            Tensor (N, C, H, W): x_t-1, the previous sample at time step t
        """
        T = T.int() # (N, )
        mean_previous = (
            self.one_minus_alphas_bar[T-1] * self.x_previous_coef[T] * x_t 
            + self.sqrt_alphas_bar[T-1] * self.x0_pred_coef[T] * x0_predicted 
        ) # (N, C, H, W)
        noise = randn_like(x0_predicted)
        return mean_previous + self.sqrt_one_minus_alphas_bar[T-1] * self.bw_std[T] * noise
        
        
    @torch.no_grad
    def sample(
        self, 
        N: int, 
        shape: int| Tuple[int] = (64, 64)
    ):
        T = self.T
        times = empty((N, ), device=self.device, dtype=torch.long).fill_(T)
        X_T = randn((N, self.img_channels) + _pair(shape), device=self.device) # (N, C, H, W)
        while T >= 1:
            X0_pred = self.forward(X_T, times) # (N, C, H, W)
            if self.reconstruction_method == "bce":
                X0_pred = sigmoid(X0_pred)
            X_T = self.reverse_q(X_T, X0_pred, times)
            T -= 1
            times -=1
        
        X_T = (X_T + 1) / 2 # scale from [-1, 1] to [0, 1]
        return [to_image(x_0) for x_0 in X_T]
        
    
    # @torch.no_grad
    # def show_translative_image(
    #     self,
    #     img_size: int | Tuple[int, int] = 256,
    #     save_path: str = None,
    # ):
    #     img_size = _pair(img_size)
    #     # Init noise to generate images from
    #     x_t = torch.randn((1, self.img_channels, *img_size), device=self.device) # [N, C, H, W]
    #
    #     plt.figure(figsize=(10, 10))
    #     n_cols = 10
    #     hidden_rows = self.T / n_cols
    #     plot_number = 1
    #     
    #     # Go from T to 0 removing and adding noise until t = 0
    #     self.model.eval()
    #     for t in range(0, self.T)[::-1]:
    #         time = torch.full((1, 1), t, device=self.device).float()
    #         e_t = self.model(x_t, time)  # Predicted noise
    #         x_t = self.reverse_q(x_t, time[0], e_t)
    #         if t % hidden_rows == 0:
    #             ax = plt.subplot(1, n_cols+1, plot_number)
    #             ax.axis('off')
    #             other_utils.show_tensor_image_diff(x_t)
    #             plot_number += 1
    #
    #     # save the image
    #     if save_path is not None:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     
    #     #plot image
    #     plt.show()


'''
class LatentDDPM(DDPM):
    def __init__(
        self,
        encoder: Type[Module],
        decoder: Type[Module],
        model: Type[Module] | None = None,
        conditional_domain_encoder: Type[Module] | None = None,
        is_conditional: bool = False,
        beta: float = 0.02,
        T: int = 100,
        device=None,
        **model_kwargs
    ):
        super().__init__(model, beta, T, device, **model_kwargs)
        
        self.encoder = encoder
        self.decoder = decoder
        if is_conditional and conditional_domain_encoder is None:
            conditional_domain_encoder = encoder
        self.conditional_domain_encoder = conditional_domain_encoder
        
        self.latent_channels = model_kwargs["img_channels"]
        self.T = T
        self.device = device
        self.is_conditional = is_conditional
        
        if model is not None:
            self.model = model
        else:
            self.model = _UNetBase(
                T=T,
                is_conditional=is_conditional,
                device=device,
                **model_kwargs
            )   
    
    
    @torch.inference_mode
    def sample(
        self, 
        latent_size: int | Tuple[int, int] | None = None,
        num_image: int = 1,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward method is the sampling process

        Args:
            latent_size (int | Tuple[int, int]): size of latent tensor.
            condition (Optional[Tensor], optional): Conditional image. Defaults to None.

        Returns:
            Tensor [N, 3, H, W]: The synthesis image
        """
        # Init noise to generate images from
        latent_size = _pair(latent_size)
        x_t = torch.randn((num_image, self.img_channels, *latent_size), device=self.device)
              
        # Process the conditional input if provided
        if condition is not None:
            self.conditional_domain_encoder.eval()
            condition: Tensor = self.conditional_domain_encoder(condition) # [N, C, H, W]
            condition = condition.flatten(start_dim=2).transpose(1, 2) # [N, H*W, C]
            condition = {
                "condition": condition,
                "need_weights": False,
            }
        else:
            condition = {}
            
        for t in range(0, self.T)[::-1]:
            time = torch.full((1, ), t, device=self.device).float()
            e_t = self.model(x_t, time, **condition)  # Predicted noise
            x_t = self.reverse_q(x_t, time, e_t)
        
        self.decoder.eval()
        return self.decoder(x_t).detach().cpu()
        
    
    @torch.inference_mode
    def show_translative_image(
        self,
        latent_size: int | Tuple[int, int] | None = 32,
        condition: Optional[Tensor] = None,
        save_path: str = None,
    ):
        # Init noise to generate images from
        if latent_size is not None: 
            latent_size = _pair(latent_size) 
        elif condition is None:
            raise ValueError("If latent_size is not specified, condition is expected to be provided")
        else:
            assert condition.dim() == 3, f"Expected condition has 3 dimensions, got {condition.dim()}"
            latent_size = condition.shape[1:] # [H, W]
            latent_size = (latent_size[0] // 8, latent_size[1] // 8)
        x_t = torch.randn((1, self.latent_channels, *latent_size), device=self.device) # [1, C, H, W]
        
        # Process the conditional input if provided
        if self.is_conditional:
            assert condition is not None, f"conditional is required, but not provided"
            self.conditional_domain_encoder.eval()
            condition = condition.unsqueeze(0) # [1, C, H, W]            
            condition = self.conditional_domain_encoder(condition) # [1, C, H, W]
            condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [1, H*W, C]
        else:
            condition = None

        plt.figure(figsize=(10, 10))
        n_cols = 10
        hidden_rows = self.T / n_cols
        plot_number = 1
        
        # Go from T to 0 removing and adding noise until t = 0
        self.decoder.eval()
        self.model.eval()
        for t in range(0, self.T)[::-1]:
            time = torch.full((1, 1), t, device=self.device).float()
            e_t = self.model(x_t, time, condition)  # Predicted noise
            x_t = self.reverse_q(x_t, time[0], e_t)
            if t % hidden_rows == 0:
                ax = plt.subplot(1, n_cols+1, plot_number)
                ax.axis('off')
                img: Tensor = self.decoder(x_t)
                other_utils.show_tensor_image_latent_diff(img)
                plot_number += 1

        # save the image
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        #plot image
        plt.show()
     

    def preprocess_mask2img(
        self,
        mask_path: str | Path,
        size: int | Tuple[int, int] = (256, 256)
    ) -> Tensor:
        
        def add_padding(image):
            h, w = image.shape[:2]

            if h > w:
                # Calculate the padding for width
                pad_width = (h - w) // 2
                pad = ((0, 0), (pad_width, pad_width))
            else:
                # Calculate the padding for height
                pad_height = (w - h) // 2
                pad = ((pad_height, pad_height), (0, 0))

            if len(image.shape) == 3:
                pad += ((0, 0), )    
            padded_image = np.pad(image, pad, mode='constant', constant_values=0)
            return padded_image
        
         # read images
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # padding
        mask = add_padding(mask)
        
        # resize images
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        
        # scale image values to [0, 1]
        mask = mask/255.0
        mask = mask.transpose(2, 0, 1)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        
        
        
        # convert to pytorch tensor
        mask = torch.from_numpy(mask)

        # scale input
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        mask = Ft.normalize(mask, mean=mean, std=std)
        
        return mask
'''



# classifier free approach - sample context noise and avg noise
@torch.no_grad()
def sample_w(
    model, ddpm, input_size, T, c, device, w_tests=None, store_freq=10
):
    if w_tests is None:
        w_tests = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    # Preprase "grid of samples" with w for rows and c for columns
    n_samples = len(w_tests) * len(c)

    # One w for each c
    w = torch.tensor(w_tests).float().repeat_interleave(len(c))
    w = w[:, None, None, None].to(device)  # Make w broadcastable
    x_t = torch.randn(n_samples, *input_size).to(device)

    # One c for each w
    c = c.repeat(len(w_tests), 1)

    # Double the batch
    c = c.repeat(2, 1)

    # Don't drop context at test time
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    x_t_store = []
    for i in range(0, T)[::-1]:
        # Duplicate t for each sample
        t = torch.tensor([i]).to(device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch
        x_t = x_t.repeat(2, 1, 1, 1)
        t = t.repeat(2, 1, 1, 1)

        # Find weighted noise
        e_t = model(x_t, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        e_t = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Deduplicate batch for reverse diffusion
        x_t = x_t[:n_samples]
        t = t[:n_samples]
        x_t = ddpm.reverse_q(x_t, t, e_t)

        # Store values for animation
        if i % store_freq == 0 or i == T or i < 10:
            x_t_store.append(x_t)

    x_t_store = torch.stack(x_t_store)
    return x_t, x_t_store