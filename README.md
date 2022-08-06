# Conditional Diffusion MNIST

This script is a minimalist implementation of a conditional diffusion model. It learns to generate MNIST digits, conditioned on a class label. The neural network architecture used is a small Unet. This code is modified from [this excellent repo](https://github.com/cloneofsimo/minDiffusion). 

The conditioning roughly follows the method described in ['Classifier-Free Diffusion Guidance'](https://arxiv.org/abs/2207.12598) (also used in [ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding'](https://arxiv.org/abs/2205.11487). Though in my experimentation, it seemed like variants of the method work, e.g. concatenating rather than the adaGroupNorm and adding embeddings seemed to work fine.

The diffusion model is from ['Denoising Diffusion Probabilistic Models (DDPM)'](https://arxiv.org/abs/2006.11239), where the added noise is predicted (rather than the cleaner image).











