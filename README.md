# **Denoising Diffusion Probabilistic Models**

[![Static Badge](https://img.shields.io/badge/Python-3.10.14-blue.svg)](https://www.python.org/) 
[![Static Badge](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://img.shields.io/badge/License-Apache2.0-red.svg)](https://github.com/quyongkeomut/Diffusion-Model/blob/main/LICENSE.md)

<!-- **Update 23/06/2025: Added gradient accumulation logic for the training stage** -->

This reposistory is a implementation of Denoising Diffusion Probabilistic Model (DDPM) implemented with PyTorch, which aim to provide a convenient and simple way to work around with Diffusion Model. By the time, I will update the Latent Diffusion Model, which is a descendant of DDPM and it works on the latent space instead of image space. Three main datasets provided are Flowers102, MNIST and FashionMNIST for a consistent comparison in the future.

## **Requirement**

- Python == 3.10.14
- PyTorch >= 2.5
- CUDA enabled computing device


## **Installation**

To install, follow these steps:

1. Clone the repository: **`$ git clone https://github.com/quyongkeomut/Diffusion-Model`**
2. Navigate to the project directory: **`$ cd Diffusion-Model`**
3. Install dependencies: **`$ pip install -r requirements.txt`**
   

## **Usage**

Base training arguments and their functionality are provided below:

```
python train_diff.py --help
usage: train_diff.py [-h] [--model MODEL] [--dataset DATASET] [--is_ddp] [--img_size IMG_SIZE] [--epochs EPOCHS] [--batch BATCH] [--lr LR] [--seed SEED]

Training args

options:
  -h, --help           show this help message and exit
  --model MODEL        Type of model, valid values are one of ['DDPM', 'LDM']
  --dataset DATASET    Dataset to train model on, valid values are one of ['flowers102', 'mnist', 'fashion_mnist']
  --is_ddp             Option for choosing training in DDP or normal training criteria
  --img_size IMG_SIZE  Image size
  --epochs EPOCHS      Num epochs
  --batch BATCH        Batch size
  --lr LR              Learning rate
  --seed SEED          Random seed for training
```
Along with base arguments, the program can accept extra arguments to provide for model/criterion/optimizer/trainer-dependent keyword arguments. Example is provided below:

```
python train_vae.py --reconstruction_method bce --is_ddp
```
In the example above, --reconstruction_method is used to decide which type of loss function to use for the reconstruction loss - in this case is BCE (Binary Cross Entropy), and is only acceptable if the target image is scaled to the range [0, 1]. To run training process, run these line of codes:

```
$ cd Diffusion-Model
$ python train_diff.py # along with extra arguments
```

For modifying the backbone of a specific Diffusion Model, you can modify the config file template corresponding to the model, which can be found at *./experiment_setup/configs* directory.
Each file has the template like this:

```yaml
NAME: "<NAME OF DIFF MODEL>"


IMG_CHANNELS: 3


LATENT_DIM: ...


BACKBONE_CONFIGS:
  stages_channels: [32, 64, 128, 256]
  expand_factor: 2
  drop_p: ...
  T: 120
  beta: 0.25
  t_embed_dim: 32
  activation: "<name of activation>"
  base_groups_norm: 4
  initializer: "<name of initializer>"
  dtype: null
  ...


OPTIMIZER_NAME: "<name of optimizer>"


OPTIM_KWARGS: 
  weight_decay: 0
  ...

```


## **Contributing**

Any contribution to the main models are welcomed. If any better model are delivered by fine-tuning the hyperparameters, or by changing the backbone, or by modifying the training 
procedure, please let me know. It would be a pleasure to include and cite your work in this repository.

If you would like to contribute your model, feel free to submit a Pull Request.

## **License**

This repository is released under the Apache License 2.9. See the **[LICENSE](https://github.com/quyongkeomut/Diffusion-Model/blob/main/LICENSE.md)** file for details.

## **Citation**

Project Title was created by **[Phuoc-Thinh Nguyen](https://github.com/quyongkeomut)**.

```
@misc{quyongkeomutVAEs,
  author = {Nguyen, P.-T.},
  title = {Diffusion Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/quyongkeomut/Diffusion-Model}}
}
```

<!--
## **Code of Conduct**

Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms. See the **[CODE_OF_CONDUCT.md](https://www.blackbox.ai/share/CODE_OF_CONDUCT.md)** file for more information.

## **FAQ**

**Q:** What is Project Title?

**A:** Project Title is a project that does something useful.

**Q:** How do I install Project Title?

**A:** Follow the installation steps in the README file.

**Q:** How do I use Project Title?

**A:** Follow the usage steps in the README file.

**Q:** How do I contribute to Project Title?

**A:** Follow the contributing guidelines in the README file.

**Q:** What license is Project Title released under?

**A:** Project Title is released under the MIT License. See the **[LICENSE](https://www.blackbox.ai/share/LICENSE)** file for details.

## **Changelog**

- **0.1.0:** Initial release
- **0.1.1:** Fixed a bug in the build process
- **0.2.0:** Added a new feature
- **0.2.1:** Fixed a bug in the new feature

## **Contact**

If you have any questions or comments about Project Title, please contact **[Your Name](you@example.com)**.

## **Conclusion**

That's it! This is a basic template for a proper README file for a general project. You can customize it to fit your needs, but make sure to include all the necessary information.
A good README file can help users understand and use your project, and it can also help attract contributors.

--!>
