import torch

from neural_nets.denoise_models.unets import BaseUNetEncoder, BaseUNetDecoder, BaseUNet
# from neural_nets.autoencoders.ae import encoder_mobilenet_v3_large, Encoder, Decoder

from torchinfo import summary


def test_GFLOPs_torchprofiler(
    model,
    model_inputs
):
    from torch.profiler import profile, record_function, ProfilerActivity
    # a couple of warm-up runs
    model(*model_inputs)
    model(*model_inputs)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            model(*model_inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def test_GFLOPs_thop(
    model,
    model_inputs
):
    from thop import profile, clever_format
    # a couple of warm-up runs
    model(*model_inputs)
    model(*model_inputs)
    macs, params = profile(model, inputs=model_inputs)
    flops, params = clever_format([macs*2, params], "%.3f")
    print("FLOPs: ", flops, "Params: ", params)
    

if __name__ == "__main__":
    
    # UNet
    model = BaseUNet(
        img_channels=3,
        stages_channels=[32, 64, 128, 256],
        latent_dim=512,
        T=100,
        t_embed_dim=128,
        is_conditional=False,
        expand_factor=3,
        drop_p=0.1,
        activation="relu",
        initializer="normal",
        device="cuda",
        dtype=None
    )  
    input_data = (
        torch.randn((1, 3, 64, 64), device="cuda"), 
        torch.tensor(1, device="cuda")
    )
    
    
    # Conditional UNet
    # model = UNetConditional(
    #      img_channels=40,
    #      down_channels=(64, 112, 128),
    #      expand_factor=2,
    #      device="cuda",
    #      kdim=40,
    #      vdim=40,
    #  ) 
    # condition = {
    #     "key": torch.randn((1, 32*32, 40), device="cuda"),
    #     "value": torch.randn((1, 32*32, 40), device="cuda"), 
    #     "need_weights": False,
    # }
    # input_data = (
    #     torch.randn((1, 40, 32, 32), device="cuda"), # representation
    #     torch.tensor(1, device="cuda"), # time
    #     condition,
    # )
    
    
    # AE Encoder
    # model = Encoder(
    #     img_channels=3,
    #     down_channels=(16, 32, 40),
    #     expand_factor=2,
    #     device="cuda"
    # )  
    # or
    # model = encoder_mobilenet_v3_large().cuda()
    # or
    # input_data = (
    #     torch.randn((1, 3, 256, 256), device="cuda"), 
    # )
    
    
    # AE Decoder
    # input_data = (
    #     torch.randn((1, 3, 64, 64), device="cuda"), 
    # )
    
    
    # model summary
    summary(model, input_data=input_data, depth=4)
    
    
    # pytorch op counter - thop
    # test_GFLOPs_thop(model, input_data)
    
    
    # torch profiler
    # test_GFLOPs_torchprofiler(model, input_data)
    
    
    ...
